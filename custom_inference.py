# Custom inference helpers for Lenovo internal deployment.
# Simulated customization on top of upstream LlamaFactory.

from dataclasses import dataclass


@dataclass
class InferenceConfig:
    """Internal inference configuration for lenovo deployment."""

    model_name_or_path: str
    adapter_name_or_path: str = ""
    max_new_tokens: int = 512
    temperature: float = 0.7
    internal_endpoint: str = "http://127.0.0.1:8000"


def load_inference_config(**kwargs) -> InferenceConfig:
    """Build an InferenceConfig from kwargs (internal helper)."""
    valid = {f.name for f in InferenceConfig.__dataclass_fields__.values()}
    return InferenceConfig(**{k: v for k, v in kwargs.items() if k in valid})


def custom_chat(model, tokenizer, messages, config: InferenceConfig) -> str:
    """Run a simple chat inference using the internal config."""
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=config.max_new_tokens,
                            temperature=config.temperature, do_sample=True)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
