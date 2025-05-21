import json
import argparse
import os
from tqdm import tqdm

from constants import GPT_JUDGE_PROMPT
from utils import get_gpt4o_answer, extract_scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--cap-file-first", type=str, default='', help='The first CHAIR jsonl file.')
    parser.add_argument("--cap-file-second", type=str, default='', help='The second CHAIR jsonl file.')

    parser.add_argument("--image-id-key", type=str, default="image_id")  # leave as default
    parser.add_argument("--caption-key", type=str, default="caption")  # leave as default

    parser.add_argument("--data-path", type=str, default="/path/to/coco/val2014/",help="data path")
    
    parser.add_argument('--api-key', type=str, required=True, default="Your Openai API key")
    parser.add_argument('--output', type=str, default='gpt_4o_response.jsonl', help='output jsonl file containing GPT responses')
    
    args = parser.parse_known_args()[0]

    assert args.cap_file_first and args.cap_file_second, 'Must input the two jsonl files!'

    caption_file_1 = []
    with open(args.cap_file_first, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                caption_file_1.append(json.loads(line))
    
    caption_file_2 = []
    with open(args.cap_file_second, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                caption_file_2.append(json.loads(line))
    
    assert len(caption_file_1) == len(caption_file_2), f'Must contains same number of captions! But got first file: {len(caption_file_1)}, second file: {len(caption_file_2)}'

    all_gpt_responses = []
    avg_hal_score_1 = 0
    avg_hal_score_2 = 0
    avg_det_score_1 = 0
    avg_det_score_2 = 0
    num_count = 0
    for idx in tqdm(range(len(caption_file_1)), total=len(caption_file_1)):
        line_1, line_2 = caption_file_1[idx], caption_file_2[idx]

        img_1, img_2 = line_1[args.image_id_key], line_2[args.image_id_key]
        assert img_1 == img_2, f'Must evaluate on same image! But for idx={idx}, image 1: {img_1}, image 2: {img_2}.'
        img_file = f"COCO_val2014_{str(img_1).zfill(12)}.jpg"
        absolute_img_path = args.data_path + os.sep + img_file

        caption_1, caption_2 = line_1[args.caption_key], line_2[args.caption_key]

        prompt = GPT_JUDGE_PROMPT.format(caption_1, caption_2)
        gpt_response = get_gpt4o_answer(prompt, absolute_img_path, api_key=args.api_key)
        print(gpt_response, '\n')

        four_scores = extract_scores(gpt_response)
        if four_scores is None:
            continue
        hal_score_1, hal_score_2, det_score_1, det_score_2 = four_scores
        
        avg_hal_score_1 += int(hal_score_1)
        avg_hal_score_2 += int(hal_score_2)
        avg_det_score_1 += int(det_score_1)
        avg_det_score_2 += int(det_score_2)
        num_count += 1

        with open(args.output, 'a', encoding='utf-8') as f:
            dict_to_dump = {
                'image_id': int(img_1),
                'baseline_caption': caption_1,
                'spin_caption': caption_2,
                'gpt_response': gpt_response
            }
            json.dump(dict_to_dump, f)
            f.write('\n')
    
    avg_hal_score_1 = float(avg_hal_score_1) / num_count
    avg_hal_score_2 = float(avg_hal_score_2) / num_count
    avg_det_score_1 = float(avg_det_score_1) / num_count
    avg_det_score_2 = float(avg_det_score_2) / num_count
    print(f"The avg hal score for Assistant 1 and Assistent 2: {avg_hal_score_1}; {avg_hal_score_2}")
    print(f"The avg det score for Assistant 1 and Assistent 2: {avg_det_score_1}; {avg_det_score_2}")
