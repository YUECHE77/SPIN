import math
import types
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


def llama_new_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    # [batch_size, num_heads, seq_len, head_dim]
    query_states = (
        self.q_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    key_states = (
        self.k_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )
    value_states = (
        self.v_proj(hidden_states)
        .view(bsz, q_len, self.num_heads, self.head_dim)
        .transpose(1, 2)
    )

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                "with a layer index."
            )
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin, position_ids
    )

    if past_key_value is not None:
        cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
        key_states, value_states = past_key_value.update(
            key_states, value_states, self.layer_idx, cache_kwargs
        )

    # [batch_size, num_heads, seq_len, seq_len] -> after first generation -> [batch_size, num_heads, 1, seq_len]
    attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

    if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
        raise ValueError(
            f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
            f" {attn_weights.size()}"
        )

    if attention_mask is not None:
        if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
            raise ValueError(
                f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
            )
        attn_weights = attn_weights + attention_mask
        attn_weights = torch.max(
            attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
        )

    # ----------------- Gating with respect to attention weights ---------------------
    if hasattr(self, "use_spin_img"):
        num_routed_head = int(self.routed_head * self.num_heads)

        attn_scores = attn_weights.permute(0, 2, 1, 3) # [B, N, H, N] or [B, 1, H, N]

        if hasattr(self, 'img_start_idx') and hasattr(self, 'img_end_idx'):
            # [B, H, N] -> [B, H]
            attn_scores_headwise = attn_scores[:, -1, :, self.img_start_idx:self.img_end_idx].sum(dim=-1).view(-1, self.num_heads)
        else:
            attn_scores_headwise = attn_scores[:,-1,:,:].sum(dim=-1).view(-1, self.num_heads)
        
        attn_score_std  = attn_scores_headwise.std(dim=1, keepdim=True)  # [B, 1]
        attn_score_norm = attn_scores_headwise / (attn_score_std / 1)  # [B, 1]
        gates = F.softmax(attn_score_norm, dim=1)  # [B, H]

        num_tokens, num_experts = gates.shape
        _, indices = torch.topk(gates, k=num_routed_head, dim=1)
        mask = F.one_hot(indices, num_classes=num_experts).sum(dim=1)  # [B, num_experts] = [B, H]

        if self.small_num_mask is not None:
            assert isinstance(self.small_num_mask, (int, float)), "small_num_mask must be a number (int or float)."
            mask = mask.to(query_states.dtype)
            mask[mask == 0] = self.small_num_mask

        if q_len > 1:
            # torch.cat([[B * (N - 1), H], [B, H]], dim=0) = [B * N, H]
            mask = torch.cat([torch.ones((bsz * (q_len-1), self.num_heads), dtype=query_states.dtype, device=query_states.device), mask], dim=0)
        
        mask = mask.reshape(bsz, q_len, -1)  # [B, N, H]
    else:
        mask = torch.ones((bsz, q_len, self.num_heads), dtype=query_states.dtype, device=query_states.device)  # [B, N, H]
    # ---------------------------------------------------------------------

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
    attn_output = torch.matmul(attn_weights, value_states)

    if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2)  # [batch_size, seq_len, num_heads, head_dim]
    # ------------- SPIN -------------
    attn_output = attn_output.reshape(bsz, q_len, self.num_heads, self.head_dim)
    attn_output = torch.einsum("bne,bned->bned", mask, attn_output)
    # -------------------------------
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def llama_modify_spin(model, start_layer, end_layer, img_start_idx, img_end_idx,
                     routed_head, use_spin_img, small_num_mask=None):
    for i in range(start_layer, end_layer):
        model.model.layers[i].self_attn.img_start_idx = img_start_idx
        model.model.layers[i].self_attn.img_end_idx = img_end_idx
        model.model.layers[i].self_attn.routed_head = routed_head
        model.model.layers[i].self_attn.use_spin_img = use_spin_img
        model.model.layers[i].self_attn.small_num_mask = small_num_mask
        model.model.layers[i].self_attn.forward = types.MethodType(llama_new_forward, model.model.layers[i].self_attn)
