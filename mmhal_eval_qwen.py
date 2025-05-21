import argparse
import json
import os
from tqdm import tqdm

import torch

from eval_data_loader import MMHalDataset
from qwen_vl.modeling_qwen import QWenLMHeadModel
from qwen_vl.qwen_generation_utils import decode_tokens, get_stop_words_ids
from qwen_vl.spin_utils import disable_torch_init, llama_modify_spin, make_context_refined  # [SPIN]

from transformers import AutoTokenizer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str, default='response_template.json', help='template file containing images and questions')
    parser.add_argument('--output', type=str, default='', help='output json file containing model responses')

    parser.add_argument("--model-path", type=str, default="/path/to/Qwen-VL-Chat/model", help="path to Qwen-VL-Chat model")

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
                        help="Leave it as default.")
    # --------------------------------------------

    args = parser.parse_args()

    assert args.batch_size == 1, f'Only tested on batch_size=1'
    disable_torch_init()

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
    # tokenizer.padding_side = 'left'

    image_processor = model.transformer.visual.image_transform
    # ------------------------------------------------------------

    # -------------- Load the dataset / dataloader ---------------
    mmhal_dataset = MMHalDataset(json_path=args.input, trans=image_processor)
    mmhal_loader = torch.utils.data.DataLoader(
        mmhal_dataset, batch_size=args.batch_size, shuffle=False, num_workers=32
    )
    # ------------------------------------------------------------

    all_results = []
    for batch_id, data in tqdm(enumerate(mmhal_loader), total=len(mmhal_loader)):
        question = data['question']
        image = data["image"]
        img_path = data["image_path"]
        line = mmhal_dataset.json_data[batch_id]

        query = tokenizer.from_list_format([
            {'image': img_path[0]}, # Either a local path or an url
            {'text': question[0]},
        ])
        
        generation_config = model.generation_config
        generation_config.repetition_penalty = args.repetition_penalty
        generation_config.temperature = 1.0  # originally 1.0
        generation_config.top_p = 1.0  # originally 0.3
        generation_config.top_k = None  # originally 0

        raw_text, input_ids, img_start_idx, img_end_idx = make_context_refined(
            tokenizer=tokenizer,
            query=query,
            history=[],  # not using history
            system='You are a helpful assistant.',
            max_window_size=generation_config.max_window_size,  # not using history
            chat_format=generation_config.chat_format,
        )

        stop_words_ids = []
        stop_words_ids.extend(get_stop_words_ids(generation_config.chat_format, tokenizer))

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
            output_attentions=False,
            output_hidden_states=False,
        )

        response = tokenizer.decode(outputs[0][input_ids.size(1):].cpu(),skip_special_tokens=True).strip()
        line['model_answer'] = response
        all_results.append(line)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
