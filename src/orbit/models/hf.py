"""Hugging Face Transformers and PEFT model client for ORBIT."""

from __future__ import annotations

import time
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from orbit.models.base import BaseModelClient, GenerationConfig, ModelOutput


class HuggingFaceModelClient(BaseModelClient):
    """Client for autoregressive generation using Hugging Face Transformers."""

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        dtype: str = "bfloat16",
        lora_adapter_path: str | None = None,
        model_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__(model_id=model_id)
        self.device_str = device
        self.dtype_str = dtype

        torch_dtype = torch.bfloat16 if dtype == "bfloat16" else torch.float32
        extra_kwargs = model_kwargs or {}

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        device_map = "auto" if device == "auto" and torch.cuda.is_available() else None

        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            **extra_kwargs,
        )

        if device != "auto" and not device_map:
            self.model.to(torch.device(device))

        if lora_adapter_path:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, lora_adapter_path)

        self.model.eval()

    def generate(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
    ) -> ModelOutput:
        cfg = config or GenerationConfig()
        start_time = time.perf_counter()

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        prompt_tokens = inputs["input_ids"].shape[1]

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": cfg.max_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }

        if cfg.temperature > 0.0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = cfg.temperature
            gen_kwargs["top_p"] = cfg.top_p
        else:
            gen_kwargs["do_sample"] = False

        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)

        generated_ids = output_ids[0][prompt_tokens:]
        completion_tokens = len(generated_ids)
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        latency = (time.perf_counter() - start_time) * 1000.0

        return ModelOutput(
            text=text,
            token_ids=generated_ids.tolist(),
            logprobs=[],  # Optional for sampling
            finish_reason="stop",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency,
            metadata={"model_id": self.model_id},
        )

    def get_logprobs(
        self,
        prompt: str,
        completion: str,
    ) -> list[float]:
        full_text = prompt + completion
        inputs = self.tokenizer(full_text, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        prompt_len = prompt_ids.shape[1]

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # [1, seq_len, vocab_size]

        # Shift logits for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = inputs["input_ids"][:, 1:]

        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs, 2, shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Slice completion portion
        completion_log_probs = token_log_probs[0, (prompt_len - 1) :].tolist()
        return completion_log_probs
