from huggingface_hub import snapshot_download
from exllamav2.model import ExLlamaV2
from exllamav2.config import ExLlamaV2Config
from exllamav2.cache import ExLlamaV2Cache
from exllamav2.tokenizer import ExLlamaV2Tokenizer
from exllamav2.generator import ExLlamaV2BaseGenerator, ExLlamaV2Sampler
import random
import os
from typing import Any, Callable
import torch


class ExLlamaV2ForCausalLM:
    def __init__(self, name: str, revision: str = "main", device="cuda:0", use_cache=True, embed_cpu=False) -> None:
        super().__init__()
        self.name = name
        self.revision = revision
        self.device = device
        self.use_cache = use_cache
        self.embed_cpu = embed_cpu

    @staticmethod
    def from_pretrained(name: str, revision="main", use_cache=True, embed_cpu=False, **kwargs):
        for k in kwargs.keys():
            print(f"WARNING: Ignoring argument `{k}` support not implemented yet.")

        device = "cuda:0"
        self = ExLlamaV2ForCausalLM(name, revision=revision, device=device, use_cache=use_cache, embed_cpu=embed_cpu)
        self.load_model()

        return self

    def load_model(self):
        model_path = f"{snapshot_download(repo_id=self.name, revision=self.revision)}/"
        assert model_path is not None

        model_directory = os.path.dirname(model_path)

        config = ExLlamaV2Config()
        config.model_dir = model_directory
        config.prepare()

        self.config = config
        self.model = ExLlamaV2(config)
        self.model.embed_cpu = self.embed_cpu
        self.model.load()
        self.tokenizer = ExLlamaV2Tokenizer(config)
        self.cache = ExLlamaV2Cache(self.model)  # create cache for inference
        self.generator = ExLlamaV2BaseGenerator(self.model, self.cache, self.tokenizer)  # create generator
        self.generator.warmup()

    def _generate_simple(
        self,
        generator,
        input_ids,
        max_new_tokens: int,
        gen_settings: ExLlamaV2Sampler.Settings,
        logits_processor: Callable,
        stopping_criteria: Callable,
    ):
        mask = None

        overflow = input_ids.shape[-1] + max_new_tokens - generator.model.config.max_seq_len
        if overflow > 0:
            input_ids = input_ids[:, -overflow:]

        generator._gen_begin_base(input_ids, mask)
        scores = []

        for i in range(max_new_tokens):
            next_token_logits = generator.model.forward(
                generator.sequence_ids[:, -1:], generator.cache, input_mask=mask
            )

            if next_token_logits.device != generator.sequence_ids.device:
                next_token_logits = next_token_logits.to(generator.sequence_ids.device)

            if logits_processor is not None:
                next_token_logits = logits_processor(
                    input_ids=generator.sequence_ids, scores=next_token_logits.squeeze(0)
                )

            next_tokens, _ = ExLlamaV2Sampler.sample(
                next_token_logits.unsqueeze(0), gen_settings, generator.sequence_ids, random.random()
            )

            generator.sequence_ids = torch.cat([generator.sequence_ids, next_tokens], dim=1)
            scores += next_token_logits.unsqueeze(0)

            if stopping_criteria is not None:
                if stopping_criteria(input_ids=generator.sequence_ids, scores=next_token_logits):
                    break

        return generator.sequence_ids, scores

    def generate(
        self,
        inputs,
        max_new_tokens=100,
        temperature=1,
        top_p=1,
        top_k=50,
        repetition_penalty=1,
        output_scores=False,
        return_dict_in_generate=False,
        logits_processor=None,
        stopping_criteria=None,
        **kwargs,
    ):
        input_ids = inputs

        # Configure generator
        settings = ExLlamaV2Sampler.Settings()
        settings.temperature = temperature
        settings.top_k = top_k
        settings.top_p = top_p
        settings.token_repetition_penalty = repetition_penalty
        settings.disallow_tokens(self.tokenizer, [])

        if settings.temperature == 0:
            # Greedy search
            settings.top_k = 1
            settings.temperature = 1

        output_ids, scores = self._generate_simple(
            self.generator, input_ids, max_new_tokens, settings, logits_processor, stopping_criteria
        )

        if return_dict_in_generate:
            output = {"sequences": output_ids}
            if output_scores:
                output["scores"] = scores
            return output
        else:
            return output_ids

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, **kwargs) -> dict[str, Any]:
        """
        Implement in subclasses of :class:`~transformers.PreTrainedModel` for custom behavior to prepare inputs in the
        generate method.
        """
        return {"input_ids": input_ids}

    def _update_model_kwargs_for_generation(
        self, outputs: Any, model_kwargs: dict[str, Any], is_encoder_decoder: bool = False
    ) -> dict[str, Any]:
        # update past
        if "past_key_values" in outputs:
            model_kwargs["past"] = outputs.past_key_values
        elif "mems" in outputs:
            model_kwargs["past"] = outputs.mems
        elif "past_buckets_states" in outputs:
            model_kwargs["past"] = outputs.past_buckets_states
        else:
            model_kwargs["past"] = None

        # update attention mask
        if not is_encoder_decoder:
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                model_kwargs["attention_mask"] = torch.cat(
                    [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        return model_kwargs

    def to(self, device):
        self.device = device
        return self
