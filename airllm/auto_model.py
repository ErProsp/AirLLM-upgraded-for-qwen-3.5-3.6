import importlib
from sys import platform
from transformers import AutoConfig, AutoModelForCausalLM

is_on_mac_os = platform == "darwin"

if is_on_mac_os:
    from airllm import AirLLMLlamaMlx


class AutoModel:
    def __init__(self):
        raise EnvironmentError(
            "AutoModel is designed to be instantiated "
            "using the `AutoModel.from_pretrained(pretrained_model_name_or_path)` method."
        )

    @classmethod
    def get_module_class(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        if "hf_token" in kwargs:
            print("using hf_token")
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=True,
                token=kwargs["hf_token"],
            )
        else:
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=True,
            )

        architectures = getattr(config, "architectures", None)
        arch = architectures[0] if architectures else ""
    
        # Qwen2
        if "Qwen2ForCausalLM" in arch:
            return "airllm", "AirLLMQWen2"

        # Qwen v1
        if "QWen" in arch:
            return "airllm", "AirLLMQWen"

        # Baichuan / ChatGLM / InternLM / Mistral / Mixtral / Llama (come originale)
        if "Baichuan" in arch:
            return "airllm", "AirLLMBaichuan"
        if "ChatGLM" in arch:
            return "airllm", "AirLLMChatGLM"
        if "InternLM" in arch:
            return "airllm", "AirLLMInternLM"
        if "Mistral" in arch:
            return "airllm", "AirLLMMistral"
        if "Mixtral" in arch:
            return "airllm", "AirLLMMixtral"
        if "Llama" in arch:
            return "airllm", "AirLLMLlama2"

        print(f"unknown architecture: {arch}, try to use Llama2...")
        return "airllm", "AirLLMLlama2"

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        if is_on_mac_os:
            return AirLLMLlamaMlx(pretrained_model_name_or_path, *inputs, **kwargs)

        # Carica la config per capire che architettura è
        cfg = AutoConfig.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=True,
            token=kwargs.get("hf_token", None),
        )

        model_type = getattr(cfg, "model_type", "")

        # === BYPASS per Qwen 3.5 / 3.6 ===
        # Qwen3.5/3.6 usa Qwen3_5DecoderLayer.forward(..., position_embeddings=...) e DynamicCache,
        # che non sono compatibili con il forward manuale di AirLLM. [page:1][web:115]
        if model_type.startswith("qwen3_5") or model_type.startswith("qwen3_6"):
            print(f"[AirLLM] Using native HuggingFace model for {model_type} (no sharding).")
            return AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                *inputs,
                trust_remote_code=True,
                **kwargs,
            )

        # === resto: usa il meccanismo AirLLM originale ===
        module_name, class_name = AutoModel.get_module_class(
            pretrained_model_name_or_path, *inputs, **kwargs
        )
        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name)
        return class_(pretrained_model_name_or_path, *inputs, **kwargs)