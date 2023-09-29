import sys, os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
)

from exllamav2.generator import ExLlamaV2BaseGenerator, ExLlamaV2Sampler

import time

# Initialize model and cache


# model_directory = "/workspace/.cache/huggingface/hub/models--TheBloke--orca_mini_v3_7B-GPTQ/snapshots/4f06a6151128861d5bb256275620f7eadcab3238/"
model_directory =  "/workspace/.cache/huggingface/hub/models--TheBloke--orca_mini_v3_70B-GPTQ/snapshots/840b85b8a347711e6219a5c6ac8a2b0f8692e995/"

config = ExLlamaV2Config()
config.model_dir = model_directory
config.embed_cpu = False
config.max_input_len = 4000
config.max_batch_size = 10
config.prepare()

model = ExLlamaV2(config)
print("Loading model: " + model_directory)

# allocate 18 GB to CUDA:0 and 24 GB to CUDA:1.
# (Call `model.load()` if using a single GPU.)
model.load()

tokenizer = ExLlamaV2Tokenizer(config)
cache = ExLlamaV2Cache(model, batch_size=2)

# Initialize generator

generator = ExLlamaV2BaseGenerator(model, cache, tokenizer)

# Generate some text

settings = ExLlamaV2Sampler.Settings()
settings.temperature = 0.8
settings.top_k = 1
settings.top_p = 0.8
settings.token_repetition_penalty = 1
settings.disallow_tokens(tokenizer, [])

txts = ["Write a story about a dog", "Write a story about a cat"]

prompts = [f"""### System:
You are a helpfull assistant with very good writing abilities.

### User:

{txt}

### Assistant:
""" for txt in txts]

max_new_tokens = 1000

generator.warmup()
time_begin = time.time()

# output = generator.generate_simple([prompt, prompt], settings, max_new_tokens, seed=1234)
output = generator.generate_simple(prompts, settings, max_new_tokens, seed=1234)

time_end = time.time()
time_total = time_end - time_begin


tokens_in = 0
for i in prompts:
    tokens_in += tokenizer.encode(i).shape[-1]

tokens_out = 0
for o in output:
    print(o)
    tokens_out += tokenizer.encode(o).shape[-1]


generated_tolkens = tokens_out - tokens_in
batch_size = len(prompts)
print(
    f"Response generated in {time_total:.2f} seconds, {batch_size} batch with input tokens {tokens_in} and {generated_tolkens} generated tokens, { generated_tolkens / time_total:.2f} tokens/second"
)
