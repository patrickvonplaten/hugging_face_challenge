# -*- coding: utf-8 -*-
from typing import List

def pad_or_clip(tokens: List[int], desired_len: int, pad_token_id: int) -> List[int]:
    """Either clips or pads a list of tokens to make it of desired length."""
    n_tokens = len(tokens)
    if n_tokens >= desired_len:
        return tokens[:desired_len]
    else:
        return tokens + [pad_token_id] * (desired_len - n_tokens)
