import argparse
import json
from tqdm import tqdm

import torch

from attentionSPIN import llama_modify_spin
from constants import INSTRUCTION_TEMPLATE, SYSTEM_MESSAGE
from llava.utils import disable_torch_init
from model_loader import ModelLoader
from eval_data_loader import MMHalDataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MMHal evaluation on LVLMs.")

    parser.add_argument('--input', type=str, default='response_template.json', help='template file containing images and questions')
    parser.add_argument('--output', type=str, default='', help='output json file containing model responses')

    parser.add_argument("--model", type=str, help="model: llava-1.5, minigpt4, shikra")
    parser.add_argument("--llava-size", type=str, default='7b', help="7b or 13b")

    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--beam", type=int, default=1)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=512)

    parser.add_argument("--start-layer", type=int, default=0)
    parser.add_argument("--end-layer", type=int, default=32)
    
    # ------------------- SPIN -------------------
    parser.add_argument("--use-spin", action="store_true", help='Use SPIN or not')
    parser.add_argument("--routed-heads", type=float, default=0.95, 
                        help='Fraction of heads that is activated for SPIN')
    parser.add_argument("--small-num-mask", type=float, default=None,
                        help="The scaling factor for SPIN")
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="Set to 1.1 when using Minigpt4")
    # --------------------------------------------

    args = parser.parse_args()

    assert args.batch_size == 1, f'Only tested on batch_size=1'
    assert args.model in ['llava-1.5', 'minigpt4', 'shikra'], (
        f'We support llava-1.5, minigpt4, and shikra. But got {args.model}. '
        'If you want to use Qwen-VL-Chat, please go to mmhal_eval_qwen.py'
    )

    disable_torch_init()

    model_loader = ModelLoader(args.model, args.llava_size)

    mmhal_dataset = MMHalDataset(json_path=args.input, trans=model_loader.image_processor)
    mmhal_loader = torch.utils.data.DataLoader(
        mmhal_dataset, batch_size=args.batch_size, shuffle=False, num_workers=32
    )

    template = INSTRUCTION_TEMPLATE[args.model]
    if args.model == "llava-1.5" or args.model == "shikra":
        template = SYSTEM_MESSAGE + template

    all_results = []
    for batch_id, data in tqdm(enumerate(mmhal_loader), total=len(mmhal_loader)):
        query = data['question']
        image = data["image"]
        line = mmhal_dataset.json_data[batch_id]

        questions, kwargs = model_loader.prepare_inputs_for_model(template, query, image)
        
        if args.use_spin:
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
                output_attentions=False,
                output_hidden_states=False,
                return_dict=True,
                **kwargs,
            )
        
        output_text = model_loader.decode(outputs)
        assert len(output_text) == 1
        line['model_answer'] = output_text[0]
        all_results.append(line)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
