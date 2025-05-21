IMAGE_TOKEN_INDEX = -200
IMAGE_TOKEN_LENGTH = 576
MINIGPT4_IMAGE_TOKEN_LENGTH = 32
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
SHIKRA_IMAGE_TOKEN_LENGTH = 256
SHIKRA_IMG_START_TOKEN = 32001
SHIKRA_IMG_END_TOKEN = 32002

INSTRUCTION_TEMPLATE = {
    "minigpt4": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "instructblip": "<ImageHere><question>",
    "lrv_instruct": "###Human: <Img><ImageHere></Img> <question> ###Assistant:",
    "shikra": "USER: <im_start><ImageHere><im_end> <question> ASSISTANT:",
    "llava-1.5": "USER: <ImageHere> <question> ASSISTANT:",
    "internvl": "USER: <ImageHere> <question> ASSISTANT:",
}

INSTRUCTION_TEMPLATE_NO_IMG = {
    "minigpt4": "###Human:<question> ###Assistant:",
    "instructblip": "<question>",
    "lrv_instruct": "###Human: <question> ###Assistant:",
    "shikra": "USER: <question> ASSISTANT:",
    "llava-1.5": "USER: <question> ASSISTANT:",
    "internvl": "USER: <question> ASSISTANT:",
}

SYSTEM_MESSAGE = "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."

POPE_CHAT_PATH = {
    "random": "./pope_coco/chat/coco_pope_chat_random.json",
    "popular": "./pope_coco/chat/coco_pope_chat_popular.json",
    "adversarial": "./pope_coco/chat/coco_pope_chat_adversarial.json",
}

# GPT-4o evaluation
GPT_JUDGE_PROMPT = '''
You are required to score the performance of two AI assistants in describing a given image. You should pay extra attention to the hallucination, which refers to the part of descriptions that are inconsistent with the image content, such as claiming the existence of something not present in the image or describing incorrectly in terms of the counts, positions, or colors of objects in the image. Please rate the responses of the assistants on a scale of 1 to 10, where a higher score indicates better performance, according to the following criteria:
1: Accuracy: whether the response is accurate with respect to the image content. Responses with fewer hallucinations should be given higher scores.
2: Detailedness: whether the response is rich in necessary details. Note that hallucinated descriptions should not count as necessary details.
Please output the scores for each criterion, containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. Following the scores, please provide an explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment.

[Assistant 1]
{}
[End of Assistant 1]

[Assistant 2]
{}
[End of Assistant 2]

Output format:
Accuracy: <Scores of the two answers>
Reason:

Detailedness: <Scores of the two answers>
Reason: 
'''
