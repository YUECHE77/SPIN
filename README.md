# Mitigating Hallucinations in Vision-Language Models through Image-Guided Head Pruning

This repository provides detailed instruction for reproducing our results reported in the paper. Other than our method SPIN, we also support PAI, DAMRO, OPERA, and VCD.

## Environment Setup

For experiments running with LLaVA-1.5 (7B or 13B), Minigpt4, and Shikra:
```
conda env create -f environment.yml
conda activate spin
```

For experiments running with Qwen-VL-Chat:
```
Please follow the official instruction of Qwen-VL to set up the environment.
```

## Prepare the Models Weights and the Scripts

Please refer to the corresponding official repositories of LLaVA-1.5, Minigpt4, Shikra, and Qwen-VL to download the model weights, and modify the necessary path (Minigpt4 and Shikra).

Go to our [ModelLoader](./model_loader.py#L171) class (under model_loader.py), and put your actual path.

Download the official MMHal Bench dataset, and modify the path in [load_image](./utils.py#158) (under utils.py) to the folder contains the MMHal Bench images.

We directly changed the source code to implement OPERA and VCD. If you also want to run experiments with those two algorithms, copy the [txt file](./transformers_utils.txt) we provide, and paste to ```transformers/generation/utils.py``` under the transformers package in your environment.

## Quick Start

Followings are the commands to run evaluation on CHAIR, POPE, as well as the MMHal Bench. If you want to try the GPT-4o evaluation, make sure to apply for an API key.

### Arguments

When you are using Qwen-VL-Chat, provide `--model-path` instead of `--model`.

| Argument           | Example         | Description                                                          |
|--------------------|-----------------|----------------------------------------------------------------------|
| `--model`          | `llava-1.5`     | Currently we support: `minigpt4`, `llava-1.5`, `shikra`.             |
| `--model-path`     | `/path/to/Qwen-VL-Chat` | Path to `Qwen-VL-Chat` model                                         |
| `--data-path`      | `/path/to/COCO` | Path to `coco/val2014/`.                                             |
| `--llava-size`     | `7b`            | To use `LLaVA-1.5-7B` or `LLaVA-1.5-13B`.                            |
| `--pope-type`      | `random`        | Type of POPE Evaluation: `random`, `popular`, or `adversarial`.      |
| `--start-layer`    | `0`             | The starting layer of applying SPIN.                                 |
| `--end-layer`      | `32`            | The ending layer of applying SPIN.                                   |
| `--use-spin`       | `-`             | Activate SPIN.                                                       |
| `--routed-heads`   | `0.95`          | Ratio of active heads (1 - ratio of suppressed heads). Default: 0.95 |
| `--small-num-mask` | `0.05`          | The scaling factor for SPIN. Default: None.                          |
| `--repetition-penalty` | `1.1`           | Set to 1.1 when using Minigpt4 on CHAIR Evaluation. Default: 1.      |
| `--output-path`    | `output.jsonl`  | Your output path.                                                    |

### 1. CHAIR Evaluation

For LLaVA-1.5, Minigpt4, and Shikra:

```bash
CUDA_VISIBLE_DEVICES=0 python chair_eval.py \
    --model llava-1.5 \
    --data-path /path/to/COCO \
    --llava-size 7b \
    --start-layer 0 \
    --end-layer 32 \
    --use-spin \
    --routed-heads 0.95 \
    --small-num-mask 0.08 \
    --repetition-penalty 1.0 \
    --output-path chair_output.jsonl
```

For Qwen-VL-Chat:

```bash
CUDA_VISIBLE_DEVICES=0 python chair_eval_qwen.py \
    --model-path /path/to/Qwen-VL-Chat \
    --data-path /path/to/COCO \
    --start-layer 0 \
    --end-layer 20 \
    --use-spin \
    --routed-heads 0.7 \
    --small-num-mask 0.08 \
    --output-path qwen_chair_output.jsonl
```

Compute the CHAIR scores with the generated jsonl file. Check `chair.py` for more detailed information. Before that, install `nltk=3.8.1`.

```bash
python chair.py --cap_file /path/to/jsonl
```

### 2. POPE Evaluation

For LLaVA-1.5, Minigpt4, and Shikra:

```bash
CUDA_VISIBLE_DEVICES=0 python pope_chat_eval.py  \
    --model llava-1.5 \
    --data-path /path/to/COCO \
    --pope-type random \
    --llava-size 7b \
    --start-layer 0 \
    --end-layer 32 \
    --use-spin \
    --routed-heads 0.8 \
    --small-num-mask 0.1 \
    --output-path pope_output_random.jsonl
```

Compute the POPE results with the generated jsonl file.

```bash
python pope_ans.py --ans-file pope_output_random.jsonl
```

### 3. MMHal Bench

For LLaVA-1.5, Minigpt4, and Shikra:

```bash
CUDA_VISIBLE_DEVICES=0 python mmhal_eval.py \
    --input /path/to/MMHal-Bench/response_template.json \
    --output mmhal_output.json \
    --model llava-1.5 \
    --llava-size 13b \
    --start-layer 0 \
    --end-layer 20 \
    --use-spin \
    --routed-heads 0.9
```

For Qwen-VL-Chat:

```bash
CUDA_VISIBLE_DEVICES=0 python mmhal_eval_qwen.py \
    --input /path/to/MMHal-Bench/response_template.json \
    --output qwen_mmhal_output.json \
    --model-path /path/to/Qwen-VL-Chat \
    --start-layer 0 \
    --end-layer 20 \
    --use-spin \
    --routed-heads 0.7 \
    --small-num-mask 0.08
```

Please follow the official instruction of MMHal-Bench to evaluate the generated responses.

### 4. GPT-4o Assisted Evaluation

Here, the caption files are the jsonl files generated by CHAIR evaluation. Please provide both of the vanilla model (first caption file) and SPIN (second caption file). You can swap the position.

```bash
python gpt_4o_eval.py \
    --cap-file-first /path/to/first/jsonl \
    --cap-file-second /path/to/second/jsonl \
    --data-path /path/to/COCO \
    --api-key your_api_key \
    --output gpt_4o_response.jsonl
```
