import random
import numpy as np
import re
import base64
import requests
import time
import os
from PIL import Image
from io import BytesIO

import torch
import torch.backends.cudnn as cudnn

def setup_seeds():
    seed = 927

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True

def add_diffusion_noise(image_tensor, noise_step):
    """
    For VCD: Adding noise to the image.
    """
    num_steps = 1000  # Number of diffusion steps

    # decide beta in each step
    betas = torch.linspace(-6,6,num_steps)
    betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5

    # decide alphas in each step
    alphas = 1 - betas
    alphas_prod = torch.cumprod(alphas, dim=0)
    alphas_prod_p = torch.cat([torch.tensor([1]).float(), alphas_prod[:-1]],0) # p for previous
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

    def q_x(x_0,t):
        noise = torch.randn_like(x_0)
        alphas_t = alphas_bar_sqrt[t]
        alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
        return (alphas_t*x_0 + alphas_1_m_t*noise)

    noise_delta = int(noise_step) # from 0-999
    noisy_image = image_tensor.clone()
    image_tensor_cd = q_x(noisy_image,noise_step) 

    return image_tensor_cd

def extract_scores(text):
    """
    GPT-4o Evaluation: Extract the scores from GPT's response.
    """
    acc_match = re.search(r'Accuracy:\s*\**\s*(\d+)\s+(\d+)', text)
    det_match = re.search(r'Detailedness:\s*\**\s*(\d+)\s+(\d+)', text)

    if not acc_match or not det_match:
        print("Failed to extract scores from GPT output.")
        return None

    hal_score_1 = int(acc_match.group(1))
    hal_score_2 = int(acc_match.group(2))
    det_score_1 = int(det_match.group(1))
    det_score_2 = int(det_match.group(2))

    return hal_score_1, hal_score_2, det_score_1, det_score_2

def get_gpt4o_answer(prompt, image_path, api_key):
    for _ in range(10):  # try 10 times maximum
        try:
            res = call_api(prompt, image_path, api_key=api_key)

            if "choices" in res:
                return res["choices"][0]["message"]["content"]
            else:
                print("API error response:", res)
                raise Exception("API returned error")

        except Exception as e:
            print("Retry due to error:", e)
            time.sleep(10)

def call_api(prompt, image_path, api_key):
    """
    GPT-4o Evaluation: Call OpenAI api.
    """

    # Function to encode the image
    def encode_image(image_path):
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Image not found: {image_path}")

    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }

    payload = {
    "model": "gpt-4o",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": prompt
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            }
        ]
        }
    ],
    "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    if response.status_code != 200:
        print(f"API returned status {response.status_code}: {response.text}")

    res_json = response.json()

    if "error" in res_json:
        print("Error from OpenAI API:", res_json["error"])
    else:
        print("Response keys:", res_json.keys())

    return res_json

def load_image(image_file):
    """
    MMHal Bench Evaluation: Read the input image.
    """
    if image_file.startswith('http') or image_file.startswith('https'):
        try:
            response = requests.get(image_file)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        except Exception as e:
            match = re.search(r'/([^/]+)\.jpg$', image_file)
            if match:
                file_id = match.group(1)
                print(f"[WARNING] Failed to load image from URL: {image_file}. Extract: {file_id}")
                local_path = os.path.join('/path/to/MMHal-Bench/images', f'{file_id}.jpg')
                image = Image.open(local_path).convert('RGB')
            else:
                raise ValueError(f'Failed to find {image_file} in local.')
    else:
        image = Image.open(image_file).convert('RGB')

    return image
