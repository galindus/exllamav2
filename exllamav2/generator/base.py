from exllamav2 import ExLlamaV2, ExLlamaV2Cache, ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2Sampler
import time
import torch
import random

import torch.nn.functional as F


class ExLlamaV2BaseGenerator:
    # Internal state

    model: ExLlamaV2
    cache: ExLlamaV2Cache
    tokenizer: ExLlamaV2Tokenizer

    sequence_ids: torch.tensor = None

    def __init__(self, model, cache, tokenizer):
        self.model = model
        self.cache = cache
        self.tokenizer = tokenizer

    # For testing purposes, run a forward pass to make sure CUDA is fully initialized

    def warmup(self):
        input_ids = torch.zeros((1, 2), dtype=torch.long)
        self.model.forward(input_ids, cache=None, input_mask=None, preprocess_only=True)

    def full(self):
        return self.sequence_ids.shape[-1] >= self.model.config.max_seq_len

    def generate_simple(self, prompt: str or list, gen_settings: ExLlamaV2Sampler.Settings, num_tokens: int, seed=None):
        if seed is not None:
            random.seed(seed)

        batch_size = 1 if isinstance(prompt, str) else len(prompt)
        ids = self.tokenizer.encode(prompt).to(self.model.device)
        mask = self.tokenizer.padding_mask(ids) if batch_size > 1 else None

        overflow = ids.shape[-1] + num_tokens - self.model.config.max_seq_len
        if overflow > 0: ids = ids[:, overflow:]

        self._gen_begin_base(ids, mask)

        forward = 0
        sample = 0
        copy = 0
        for i in range(num_tokens):
            start = time.time()
            logits = self.model.forward(self.sequence_ids[:, -1:], self.cache, input_mask=mask)
            fw = time.time()
            token, _ = ExLlamaV2Sampler.sample(logits, gen_settings, self.sequence_ids, random.random())
            s = time.time()
            token = token.to(self.sequence_ids.device)
            self.sequence_ids = torch.cat([self.sequence_ids, token], dim=1)
            cp = time.time()
            forward +=  fw - start
            sample += s - fw
            copy += cp - s

        print(f"forward: {forward}, sample: {sample}, copy: {copy}")
        text = self.tokenizer.decode(self.sequence_ids)

        if isinstance(prompt, str):
            return text[0]
        return text

    def _gen_begin_base(self, input_ids, mask=None):
        if self.cache:
            self.cache.current_seq_len = 0

        self.model.forward(input_ids[:, :-1], self.cache, input_mask=mask, preprocess_only=True)

        self.sequence_ids = input_ids.clone()
        self.sequence_ids = input_ids
