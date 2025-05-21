import argparse
import json
from tqdm import tqdm
import numpy as np
import torch

from attentionPAI import llama_modify
from attentionSPIN import llama_modify_spin

from constants import (
    INSTRUCTION_TEMPLATE,
    INSTRUCTION_TEMPLATE_NO_IMG,
    POPE_CHAT_PATH,
    SYSTEM_MESSAGE,
)
from eval_data_loader import POPEChatDataSet
from llava.utils import disable_torch_init
from model_loader import ModelLoader

from CFG_damro import DamroCFGLogits
from utils import setup_seeds, add_diffusion_noise

from transformers.generation.logits_process import LogitsProcessorList

parser = argparse.ArgumentParser(description="POPE chat evaluation on LVLMs.")
parser.add_argument("--model", type=str, help="model: llava-1.5, minigpt4, shikra")
parser.add_argument("--pope-type", type=str, help="random, popular, or adversarial")
parser.add_argument(
    "--data-path",
    type=str,
    default="/path/to/coco/val2014/",
    help="data path",
)

parser.add_argument("--llava-size", type=str, default='7b', help="7b or 13b")

parser.add_argument("--batch-size", type=int, default=1)
parser.add_argument("--beam", type=int, default=1)
parser.add_argument("--sample", action="store_true")
parser.add_argument("--max-tokens", type=int, default=512)

parser.add_argument("--start-layer", type=int, default=0)
parser.add_argument("--end-layer", type=int, default=32)

# -------------------- PAI -------------------
parser.add_argument("--use-pai", action="store_true", help='Use PAI or not')
parser.add_argument("--use-attn", action="store_true")
parser.add_argument("--alpha", type=float, default=0.2)
parser.add_argument("--use-cfg", action="store_true")
parser.add_argument("--gamma", type=float, default=2)
# --------------------------------------------

# ------------------- SPIN -------------------
parser.add_argument("--use-spin", action="store_true", help='Use SPIN or not')
parser.add_argument("--routed-heads", type=float, default=0.95, 
                    help='Fraction of heads that is activated for SPIN')
parser.add_argument("--small-num-mask", type=float, default=None,
                    help="The scaling factor for SPIN")
parser.add_argument("--repetition-penalty", type=float, default=1.0,
                    help="Leave as default.")
# --------------------------------------------

# ---------------- DAMRO ----------------
parser.add_argument("--use-damro", action="store_true", help='Use DAMRO or not')
parser.add_argument("--alpha-damro", type=float, default=0.2,
                    help='alpha for DAMRO logits refinement')
parser.add_argument("--outlier-topk", type=int, default=10,
                    help='Number of outliers to be removed')
# --------------------------------------

# ------------------- OPERA -------------------
parser.add_argument("--use-opera", action="store_true", help='Use OPERA or not')
parser.add_argument("--scale-factor", type=float, default=50.0)
parser.add_argument("--threshold", type=int, default=15)
parser.add_argument("--num-candidates", type=int, default=5)
parser.add_argument("--penalty-weights", type=float, default=1.0)
# ---------------------------------------------

# ------------------- VCD -------------------
parser.add_argument("--use-cd", action='store_true', help='Use VCD or not')
parser.add_argument("--noise-step", type=int, default=999)
parser.add_argument("--cd-alpha", type=float, default=1.0)
parser.add_argument("--cd-beta", type=float, default=0.1)
# -------------------------------------------

parser.add_argument("--output-path", type=str, default="", help="saving POPE results.")
args = parser.parse_known_args()[0]

assert args.model in ['llava-1.5', 'minigpt4', 'shikra'], (
    f'We support llava-1.5, minigpt4, and shikra. But got {args.model}. '
    'If you want to use Qwen-VL-Chat, please go to pope_chat_eval_qwen.py'
)
if not args.output_path:
    print('You are running the script without saving the outputs.')
if args.model == 'minigpt4' and args.use_cd:
    raise ValueError('VCD is not implemented for Q-former based model like Minigpt4.')
if args.model != 'llava-1.5' and args.use_damro:
    raise ValueError('DAMRO only implemented for llava.')

setup_seeds()
disable_torch_init()

model_loader = ModelLoader(args.model, args.llava_size)

args.pope_path = POPE_CHAT_PATH[args.pope_type]
pope_dataset = POPEChatDataSet(
    pope_path=args.pope_path,
    data_path=args.data_path,
    trans=model_loader.image_processor,
)
pope_loader = torch.utils.data.DataLoader(
    pope_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=32,
    drop_last=False,
)

template = INSTRUCTION_TEMPLATE[args.model]
if args.model == "llava-1.5" or args.model == "shikra":
    template = SYSTEM_MESSAGE + template

template_no_img = INSTRUCTION_TEMPLATE_NO_IMG[args.model]

for batch_id, data in tqdm(enumerate(pope_loader), total=len(pope_loader)):
    image = data["image"]
    queries = np.array(data["query"])
    label = torch.stack(data["label"])
    kwargs = {}
    history = None

    round = label.size()[0]

    for idx in range(round):
        query = queries[idx, :].tolist()
        lal = label[idx, :].tolist()
        # prepare inputs for model
        if history is None:
            questions, kwargs = model_loader.prepare_inputs_for_model(
                template, query, image
            )
        else:
            # Default bs is 1
            history_tmp = [his + template_no_img for his in history][0]
            questions, kwargs = model_loader.prepare_inputs_for_model(
                history_tmp, query, image
            )
        history = questions

        # ------------------- VCD -------------------
        if args.use_cd:
            image_tensor_cd = add_diffusion_noise(kwargs["images"], args.noise_step)
            kwargs['images_cd'] = image_tensor_cd
            kwargs['cd_alpha'] = args.cd_alpha
            kwargs['cd_beta'] = args.cd_beta
        else:
            image_tensor_cd = None
        # -------------------------------------------

        # ------------------- OPERA -----------------
        if args.use_opera:
            key_position = {
                "image_start": model_loader.img_start_idx, 
                "image_end": model_loader.img_end_idx - 1,
                "response_start": model_loader.response_start_idx,
            }
            kwargs['opera_decoding'] = True
            kwargs['key_position'] = key_position
            kwargs['scale_factor'] = args.scale_factor
            kwargs['threshold'] = args.threshold
            kwargs['num_attn_candidates'] = args.num_candidates
            kwargs['penalty_weights'] = args.penalty_weights
            output_attentions = True  # Optional. But required for OPERA
        else:
            kwargs['opera_decoding'] = False
            kwargs['key_position'] = None
            output_attentions = False  # You can set it to True if you want
        # -------------------------------------------

        if args.use_pai:
            llama_modify(
                model_loader.llm_model,
                args.start_layer,
                args.end_layer,
                args.use_attn,
                args.alpha,
                args.use_cfg,
                model_loader.img_start_idx,
                model_loader.img_end_idx,
            )

            logits_processor = (
                model_loader.init_cfg_processor(questions, args.gamma, args.beam, args.start_layer, args.end_layer)
                if args.use_cfg
                else None
            )

            if logits_processor is not None:
                kwargs["logits_processor"] = LogitsProcessorList([logits_processor])

        elif args.use_spin:
            llama_modify_spin(
                model_loader.llm_model,
                args.start_layer,
                args.end_layer,
                model_loader.img_start_idx,
                model_loader.img_end_idx,
                routed_head=args.routed_heads,
                use_spin_img=True,
                small_num_mask=args.small_num_mask,
            )
        
        elif args.use_damro:
            logits_processor = DamroCFGLogits(args.alpha_damro, model_loader.llm_model, images=kwargs["images"])
            kwargs["logits_processor"] = LogitsProcessorList([logits_processor])

        with torch.inference_mode():
            model_loader.llm_model.generation_config.repetition_penalty = args.repetition_penalty
            model_loader.llm_model.generation_config.temperature = 1.0  # originally 1.0
            model_loader.llm_model.generation_config.top_p = 1.0  # originally 1.0
            model_loader.llm_model.generation_config.top_k = None  # originally 50
            
            outputs = model_loader.llm_model.generate(
                do_sample=args.sample,
                max_new_tokens=args.max_tokens,
                use_cache=True,
                num_beams=args.beam,
                output_attentions=output_attentions,
                output_hidden_states=False,
                return_dict=True,
                **kwargs,
            )

        output_text = model_loader.decode(outputs)

        history = [
            h + o + model_loader.tokenizer.eos_token
            for h, o in zip(history, output_text)
        ]

        if args.output_path:
            for i in range(len(output_text)):
                with open(args.output_path, "a", encoding='utf-8') as f:
                    json.dump(
                        {
                            "query": query[i],
                            "label": lal[i],
                            "ans": output_text[i],
                            "question": questions[i],
                        },
                        f,
                    )
                    f.write("\n")
