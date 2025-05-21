import torch
import torch.nn.functional as F
from transformers import (
    LogitsProcessor,
)


class DamroCFGLogits(LogitsProcessor):
    def __init__(
        self,
        alpha,
        model,
        images=None,
        input_type="inputs_ids",
    ):
        self.alpha = alpha
        self.input_type = input_type
        self.model = model
        self.images = images
        self.out = None

    def __call__(self, input_ids, scores):
        scores = F.log_softmax(scores, dim=-1)
        if self.alpha == 1:
            return scores

        if self.out is None: # get logits only for text tokens
            if self.input_type == "inputs_ids":
                self.out = self.model(input_ids, images=self.images, use_cache=True, outlier_tokens=True)
            else:
                print("Neither input_ids nor inputs_embeds is provided.")
        else:
            self.out = self.model(
                input_ids[:, -1:],
                use_cache=True,
                past_key_values=self.out.past_key_values,
            )

        unconditional_logits = F.log_softmax(self.out.logits[:, -1, :], dim=-1)

        # cutoff = torch.log(torch.tensor(0.1)) + scores.max(dim=-1, keepdim=True).values
        cutoff = scores.max(dim=-1, keepdim=True).values

        out = (
            self.alpha * (scores - unconditional_logits) + unconditional_logits
        )
        cd_logits = out.masked_fill(scores < cutoff, -float("inf"))
        return cd_logits
