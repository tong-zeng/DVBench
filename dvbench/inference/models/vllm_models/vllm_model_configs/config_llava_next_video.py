from pydantic import Field
from typing import Optional
from dvbench.inference.models.base import (
    ModelConfig,
    ModelInitConfig,
    ModelSamplingConfig,
)
from dvbench.inference.configs.app import (
    GLOBAL_INIT,
    GLOBAL_SAMPLING,
)


class LLaVANextVideovLLMInitConfig(ModelInitConfig):
    pretrained_model_name_or_path: str = "llava-hf/LLaVA-NeXT-Video-7B-hf"
    max_model_len: int = GLOBAL_INIT.max_model_len
    trust_remote_code: bool = GLOBAL_INIT.trust_remote_code
    gpu_memory_utilization: float = GLOBAL_INIT.gpu_memory_utilization
    quantization: Optional[str] = GLOBAL_INIT.quantization
    max_num_frames: int = GLOBAL_INIT.max_num_frames
    torch_dtype: str = "auto"
    device: str = "cuda"
    use_fast: bool = GLOBAL_INIT.use_fast
    target_fps: int = GLOBAL_INIT.target_fps


class LLaVANextVideovLLMSamplingConfig(ModelSamplingConfig):
    temperature: float = GLOBAL_SAMPLING.temperature
    top_p: float = GLOBAL_SAMPLING.top_p
    top_k: int = GLOBAL_SAMPLING.top_k
    max_tokens: int = GLOBAL_SAMPLING.max_tokens
    # use_beam_search: bool = False # beam search is not supported in this version of vLLM, expecially for VLLM
    # best_of: int = GLOBAL_SAMPLING.best_of # use best_of is used toget ther with the beam search, being treated as the beam width. "n must be 1 when using greedy sampling"


class LLaVANextVideovLLMModelConfig(ModelConfig):
    init_config: LLaVANextVideovLLMInitConfig = Field(
        default_factory=LLaVANextVideovLLMInitConfig
    )
    sampling_config: LLaVANextVideovLLMSamplingConfig = Field(
        default_factory=LLaVANextVideovLLMSamplingConfig
    )
