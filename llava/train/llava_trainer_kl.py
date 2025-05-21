import os
import logging

import torch
from torch.utils.data import Sampler

from transformers import Trainer
from transformers.trainer import (
    has_length,
)
from typing import List, Optional
from copy import deepcopy

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX
from transformers.modeling_utils import unwrap_model
from transformers.trainer import _is_peft_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    assert len(mm_indices) > 0, "Should have at least one multimodal sample."
    assert len(lang_indices) > 0, "Should have at least one language sample."

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) >= megabatch_size:
        megabatches = [additional_batch[:megabatch_size]] + megabatches
        additional_batch = additional_batch[megabatch_size:]

    if len(additional_batch) > 0:
        megabatches.append(additional_batch)

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)


class ShapeMismatchError(Exception):
    def __init__(self, iteration, teacher_shape, student_shape):
        super().__init__(f"Batch number {iteration}: pd of teacher and student have different shape. Got teacher: {teacher_shape}, student: {student_shape}")
        self.iteration = iteration
        self.teacher_shape = teacher_shape
        self.student_shape = student_shape


class LLaVATrainer(Trainer):

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        if self.args.group_by_modality_length:
            lengths = self.train_dataset.modality_lengths
            return LengthGroupedSampler(
                # self.args.train_batch_size * self.args.gradient_accumulation_steps, # TODO: seems that we should not have gradient_accumulation_steps
                self.args.train_batch_size,
                world_size=self.args.world_size,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def _save_checkpoint(self, model, trial, metrics=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', 'vision_resampler']
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            # also save pos embedding
            if getattr(self.args, "tune_vit_pos_embedding", False):
                keys_to_match.extend(['vision_tower.embeddings.position_embedding'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)
            print("weight to save:", weight_to_save.keys())

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(LLaVATrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(LLaVATrainer, self)._save(output_dir, state_dict)
    
    @staticmethod
    def shift_img_idx(indices):
        """
        Get the correct indices for the image replacement tokens
        correct index: The index that all the '<image>' before it are expanded
        This function is very important, and the logic here is complicated -> see your draft carefully
        """
        if indices.numel() == 0:
            return torch.tensor([])  # I want consistency

        image_len = 576  # 576 is the fixed image length for llava -> (336 / 14) * (336 / 14) = 576
        shifts = torch.arange(0, len(indices)) * image_len - torch.arange(0, len(indices))

        return indices + shifts.to(indices.device)
    
    def prepare_no_image_inputs(self, inputs):
        """
        Edge case: Within a batch, number of images for each instance might be different -> need padding
        Number of images might be: 0, 1, more than 1
        In normal case, the longest sequence doesn't need padding
        But after dropping the Image token <image>, to keep the same dimension, we still need to pad. Or do we?
        """
        new_inputs = {}
        new_input_ids = []
        new_labels = []
        new_attention_mask = []

        all_image_tokens_index = []  # the indices for all the <image> tokens (for each example in the batch)

        assert inputs['input_ids'].shape == inputs['attention_mask'].shape == inputs['labels'].shape, 'The shapes are different in inputs.'
        batch_size, _ = inputs['input_ids'].shape

        position_mask = (inputs['input_ids'] != IMAGE_TOKEN_INDEX)  # [batch_size, max_len_in_batch] -> True / False
        
        for b in range(batch_size):
            row_mask = position_mask[b]

            row_input_ids = inputs['input_ids'][b][row_mask]  # [seq_len, ]
            row_labels = inputs['labels'][b][row_mask]  # [seq_len, ]
            row_attention_mask = inputs['attention_mask'][b][row_mask]  # [seq_len, ]

            new_input_ids.append(row_input_ids)
            new_attention_mask.append(row_attention_mask)
            new_labels.append(row_labels)

            indices = (inputs['input_ids'][b] == IMAGE_TOKEN_INDEX).nonzero(as_tuple=True)[0]
            shifted_indices = self.shift_img_idx(indices)  # if no image -> empty tensor
            all_image_tokens_index.append(shifted_indices)  # [[idx_1, idx_2], [idx_1], [], ..., [idx_1, idx_2, idx_3]]
        
        # self.tokenizer.model_max_length = 2048
        new_inputs['input_ids'] = torch.nn.utils.rnn.pad_sequence(new_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)[:, :self.tokenizer.model_max_length]
        new_inputs['labels'] = torch.nn.utils.rnn.pad_sequence(new_labels, batch_first=True, padding_value=IGNORE_INDEX)[:, :self.tokenizer.model_max_length]
        new_inputs['attention_mask'] = new_inputs['input_ids'].ne(self.tokenizer.pad_token_id)
        new_inputs['images'] = None

        all_valid_len = torch.sum(new_inputs['attention_mask'], dim=1)  # number of paddings in each example

        assert new_inputs['input_ids'].shape == new_inputs['labels'].shape == new_inputs['attention_mask'].shape, 'The shapes are different in new_inputs (after removal).'

        return new_inputs, all_image_tokens_index, all_valid_len
    
    def KL_loss(self, with_image_logits, without_image_logits, all_image_tokens_index, all_valid_len, gamma=1.2, temperature=1.0, scale=False):
        """
        Inspired by PAI:
        The teacher should be: p_model = gamma * p_with_img - (gamma - 1) * p_without_img
        The student should be: p_with_img

        About with_image_logits and without_image_logits
        Their shape: [batch_size, seq_len, vocab_size] -> but not the same
        e.g. with_image_logits      = [3, 691, 32000] = [3, 116 - 1 + 576, 32000]
             without_image_logits   = [3, 115, 32000] = [3, 116 - 1, 32000]
        """
        assert with_image_logits.shape[0] == without_image_logits.shape[0] and with_image_logits.shape[-1] == without_image_logits.shape[-1], 'The batch size or (and) vocab_size does not match.'
        batch_size = with_image_logits.shape[0]

        p_with_img = torch.nn.functional.log_softmax(with_image_logits / temperature, dim=-1)  # It is also student pd
        p_without_img = torch.nn.functional.log_softmax(without_image_logits / temperature, dim=-1)

        image_len = 576  # 576 is the fixed image length for llava -> (336 / 14) * (336 / 14) = 576
        kl_divergence = 0
        for b in range(batch_size):
            valid_len = all_valid_len[b]  # a scaler -> number of text tokens
            image_pos = all_image_tokens_index[b]  # a tensor -> [idx_1, idx_2, ..., idx_n]

            if (valid_len + image_len * len(image_pos)) >= self.tokenizer.model_max_length and image_pos.numel() > 0:
                # that means it has been truncated in prepare_inputs_labels_for_multimodal()
                length_to_add = (torch.arange(len(image_pos)) + 1) * image_len  # how many tokens will be added if expand
                exceed_idx = torch.nonzero(length_to_add.to(valid_len.device) + valid_len >= self.tokenizer.model_max_length, as_tuple=True)[0].item()  # what if when expanding to this image, the length is already exceeded
                image_pos = image_pos[:exceed_idx + 1]  # valid images
                
                # two possibilities: the exceeding part is after this image (image_pos[exceed_idx]), or in the middle of it
                if image_pos[-1] + image_len <= self.tokenizer.model_max_length:
                    # that means the first possibility -> the exceeding part is after this image
                    total_text_token_num = self.tokenizer.model_max_length - (len(image_pos) * image_len)
                else:
                    # that means the second possibility -> the exceeding part is in the middle of expanded image token
                    # under this circumstance, when computing teacher_row_pd -> teacher_row_pd[image_pos[-1] + image_len:, :], 
                    # where "image_pos[-1] + image_len" is greater than self.tokenizer.model_max_length. But this won't cause bugs.
                    total_text_token_num = image_pos[-1] - ((len(image_pos) - 1) * image_len)

                single_p_without_img = p_without_img[b][:total_text_token_num, :]  # [total_text_token_num, vocab_size]
                single_p_with_img = p_with_img[b]  # [self.tokenizer.model_max_length, vocab_size] = [2048, 32000]
            else:
                single_p_without_img = p_without_img[b][:valid_len, :]  # [valid_len, vocab_size] -> get rid of the padding
                single_p_with_img = p_with_img[b][:valid_len + image_len * len(image_pos), :]
            
            if image_pos.numel() > 0:
                temp = [single_p_with_img[image_pos[idx] + image_len : image_pos[idx + 1], :] for idx in range(len(image_pos) - 1)]
                single_p_with_img = torch.cat([single_p_with_img[0 : image_pos[0], :], ] + temp + [single_p_with_img[image_pos[-1] + image_len:, :], ], dim=0) 

            if single_p_without_img.shape != single_p_with_img.shape:
                raise ShapeMismatchError(b, single_p_with_img.shape, single_p_without_img.shape)
            
            # Exactly the same as PAI
            cutoff = torch.log(torch.tensor(0.1)) + single_p_with_img.max(dim=-1, keepdim=True).values
            p_model = gamma * single_p_with_img - (gamma - 1) * single_p_without_img
            cd_logits = p_model.masked_fill(single_p_with_img < cutoff, -float('inf'))  # [seq_len, vocab_size]
            ##########################

            teacher_pd = torch.nn.functional.softmax(cd_logits / temperature, dim=-1)
            student_pd = single_p_with_img
            kl_divergence += torch.nn.functional.kl_div(input=student_pd, target=teacher_pd, reduction='batchmean')  # take average over seq_len
        
        kl_divergence /= batch_size  # take average over batch

        if scale:
            kl_divergence = kl_divergence * (temperature ** 2)  # Hinton et al., 2015
        
        return kl_divergence
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss function -> distillation: with and without image
        Can also override training_step() for other behaviours
        Print the sentence: self.tokenizer.decode(inputs['input_ids'][0][inputs['input_ids'][0] > 0])

        inputs["input_ids"].shape = inputs["labels"].shape = inputs["attention_mask"].shape = [batch_size, max_len_in_batch]. e.g. [3, 116]
        inputs["images"].shape = [batch_size, 3, 336, 336]
        """
        # remove all the image replacement values in input_ids and set images to None -> without image
        inputs_copy = deepcopy(inputs)
        inputs_no_image, all_image_tokens_index, all_valid_len = self.prepare_no_image_inputs(inputs_copy)

        if self.label_smoother is not None and 'labels' in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        
        # forward pass
        # outputs.logits.shape = [batch_size, max_len_in_batch, vocab_size] = [3, 691, 32000] = [3, 116 - 1 + 576, 32000]
        outputs = model(**inputs)
        outputs_no_image = model(**inputs_no_image)  # outputs_no_image.logits.shape = [3, 115, 32000]

        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        
        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()

            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]  # The cross entropy loss here
        
        # --------------------------- Comes to the distillation part ---------------------------
        temperature = 2.0
        scaler = 0.7  # maybe do not use the scaler
        gamma = 1.2  # gamma from PAI

        try:
            kl_divergence = self.KL_loss(outputs["logits"],
                                         outputs_no_image["logits"],
                                         all_image_tokens_index=all_image_tokens_index,
                                         all_valid_len=all_valid_len,
                                         gamma=gamma,
                                         temperature=temperature,
                                         scale=True)
        except ShapeMismatchError as e:
            print(("\n\n-----The mismatched sentence-----\n\n"
                   f"{self.tokenizer.decode(inputs['input_ids'][e.iteration][inputs['input_ids'][e.iteration] > 0])}\n\n"))
            logging.error("Shape mismatch error occurred", exc_info=True)
            raise
        
        final_loss = scaler * kl_divergence + (1 - scaler) * loss
        # -------------------------------------- end --------------------------------------
        return (final_loss, outputs) if return_outputs else final_loss
