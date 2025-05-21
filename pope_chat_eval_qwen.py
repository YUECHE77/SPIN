import argparse
import json
import os
import numpy as np
from tqdm import tqdm

import torch
import torch.utils
import torch.utils.data

from eval_data_loader import POPEChatDataSet
from utils import setup_seeds
from constants import POPE_CHAT_PATH

from qwen_vl.modeling_qwen import QWenLMHeadModel
from qwen_vl.qwen_generation_utils import decode_tokens, get_stop_words_ids
from qwen_vl.spin_utils import disable_torch_init, llama_modify_spin, make_context_refined  # [SPIN]
from qwen_vl.spin_utils import pai_llama_modify, init_cfg_processor  # [PAI]
from qwen_vl.spin_utils import add_diffusion_noise, evolve_vcd_sampling  # [VCD]

from transformers import AutoTokenizer
from transformers.generation.logits_process import LogitsProcessorList

parser = argparse.ArgumentParser(description="POPE chat evaluation on Qwen-VL-Chat.")
parser.add_argument("--model-path", type=str, default="/path/to/Qwen-VL-Chat/model", help="path to Qwen-VL model")
parser.add_argument("--pope-type", type=str, default="random", help="random, popular, or adversarial")
parser.add_argument("--data-path", type=str, default="/path/to/coco/val2014/", help="data path")

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
parser.add_argument("--use-mask", action="store_true")
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

parser.add_argument("--output-path", type=str, default="", help="saving POPE chat results.")
args = parser.parse_known_args()[0]

assert args.batch_size == 1, 'Currently only support batch size = 1'
if not args.output_path:
    print('You are running the script without saving the outputs.')

setup_seeds()
disable_torch_init()
evolve_vcd_sampling()  # [VCD]

# --------- Load Model, Tokenizer, and Img Processor ---------
model_path = os.path.expanduser(args.model_path)
model = QWenLMHeadModel.from_pretrained(
    model_path,
    device_map='auto',
    trust_remote_code=True, 
    fp16=True,
).eval()

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token_id = tokenizer.eod_id  # you can also set bos_id and eos_id to tokenizer.eod_id
# tokenizer.padding_side = 'left'  # VCD

image_processor = model.transformer.visual.image_transform
# ------------------------------------------------------------

# -------------- Load the dataset / dataloader ---------------
pope_path = POPE_CHAT_PATH[args.pope_type]
pope_dataset = POPEChatDataSet(
    pope_path=pope_path,
    data_path=args.data_path,
    trans=image_processor,
)

pope_loader = torch.utils.data.DataLoader(
    pope_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=32,
    drop_last=False,
)
# ------------------------------------------------------------

for batch_id, data in tqdm(enumerate(pope_loader), total=len(pope_loader)):
    image = data["image"]
    img_path = data["image_path"]
    queries = np.array(data["query"])
    label = torch.stack(data["label"])

    kwargs = {}
    history = []
    history_no_img = []
    round = label.size()[0]

    generation_config = model.generation_config
    generation_config.repetition_penalty = args.repetition_penalty
    generation_config.temperature = 1.0  # originally 1.0
    generation_config.top_p = 1.0  # originally 0.3
    generation_config.top_k = None  # originally 0

    # ------------------- VCD -------------------
    if args.use_cd:
        image_tensor_cd = add_diffusion_noise(image, args.noise_step)
    else:
        image_tensor_cd = None
    # -------------------------------------------
    
    stop_words_ids = []
    stop_words_ids.extend(get_stop_words_ids(generation_config.chat_format, tokenizer))

    for idx in range(round):
        question = queries[idx, 0].item()
        lal = label[idx, 0]

        if len(history) == 0:
            query = tokenizer.from_list_format([
                {'image': img_path[0]},  # Either a local path or an url
                {'text': question},
            ])

            query_no_img = tokenizer.from_list_format([
                {'text': question},
            ])
        else:
            query = query_no_img = question
        
        raw_text, input_ids, img_start_idx, img_end_idx = make_context_refined(
            tokenizer=tokenizer,
            query=query,
            history=history,
            system='You are a helpful assistant.',
            max_window_size=generation_config.max_window_size,  # maximum history length
            chat_format=generation_config.chat_format,
        )

        # ------------------ OPERA ------------------
        if args.use_opera:
            key_position = {
                "image_start": img_start_idx, 
                "image_end": img_end_idx - 1,
                "response_start": input_ids.shape[1],
            }
            output_attentions = True  # Optional. But required for OPERA
        else:
            key_position = None
            output_attentions = False  # you can also set it to True if you like
        # -------------------------------------------

        logits_processor_list = None
        if args.use_spin:
            llama_modify_spin(
                model=model,
                start_layer=args.start_layer,
                end_layer=args.end_layer,
                img_start_idx=img_start_idx,
                img_end_idx=img_end_idx,
                routed_head=args.routed_heads,
                use_spin_img=True,
                small_num_mask=args.small_num_mask,
            )
        
        elif args.use_pai:
            pai_llama_modify(
                model=model,
                start_layer=args.start_layer,
                end_layer=args.end_layer,
                img_start_idx=img_start_idx,
                img_end_idx=img_end_idx,
                use_attn=args.use_attn, 
                alpha=args.alpha, 
                use_cfg=args.use_cfg,
            )

            if args.use_cfg:
                logits_processor = init_cfg_processor(query_no_img, 
                                                      tokenizer=tokenizer,
                                                      model=model,
                                                      gamma=args.gamma, 
                                                      beam=args.beam, 
                                                      start_layer=args.start_layer, 
                                                      end_layer=args.end_layer,
                                                      device=input_ids.device,
                                                      history=history_no_img,
                                                      system='You are a helpful assistant.',
                                                      max_window_size=generation_config.max_window_size,
                                                      chat_format=generation_config.chat_format,)

                logits_processor_list = LogitsProcessorList([logits_processor])
        
        outputs = model.generate(
            input_ids,  # input_ids
            images=image,  # image_tensor
            do_sample=args.sample,
            num_beams=args.beam,
            max_new_tokens=args.max_tokens,
            min_new_tokens=1,
            use_cache=True,
            pad_token_id=tokenizer.eod_id,
            eos_token_id=tokenizer.eod_id,
            stop_words_ids=stop_words_ids,
            return_dict_in_generate=False,
            generation_config=generation_config,
            output_attentions=output_attentions,
            output_hidden_states=False,
            logits_processor=logits_processor_list,  # [PAI]
            images_cd=(image_tensor_cd if image_tensor_cd is not None else None),  # [VCD]
            cd_alpha = args.cd_alpha,  # [VCD]
            cd_beta = args.cd_beta,  # [VCD]
            opera_decoding=args.use_opera,  # [OPERA]
            key_position=key_position,  # [OPERA]
            scale_factor=args.scale_factor,  # [OPERA]
            threshold=args.threshold,  # [OPERA]
            num_attn_candidates=args.num_candidates,  # [OPERA]
            penalty_weights=args.penalty_weights,  # [OPERA]
        )

        response = tokenizer.decode(outputs[0][input_ids.size(1):].cpu(),skip_special_tokens=True).strip()
        history.append((query, response))
        history_no_img.append((query_no_img, response))

        if args.output_path:
            with open(args.output_path, "a", encoding='utf-8') as f:
                dict_to_dump = {
                    "query": question,
                    "label": int(lal.item()),
                    "ans": response,
                    "question": raw_text,
                }
                json.dump(dict_to_dump, f)
                f.write("\n")
