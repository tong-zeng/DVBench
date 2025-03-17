from pydantic import Field
from typing import Optional
from driving_vllm_arena.inference.models.base import (
    ModelInitConfig,
    ModelSamplingConfig,
    ModelConfig,
)
from driving_vllm_arena.inference.configs.app import (
    GLOBAL_INIT,
    GLOBAL_SAMPLING,
)


class LLaVAOneVisionvLLMInitConfig(ModelInitConfig):
    pretrained_model_name_or_path: str = "llava-hf/llava-onevision-qwen2-7b-ov-hf"
    max_model_len: int = GLOBAL_INIT.max_model_len
    trust_remote_code: bool = GLOBAL_INIT.trust_remote_code
    gpu_memory_utilization: float = GLOBAL_INIT.gpu_memory_utilization
    quantization: Optional[str] = GLOBAL_INIT.quantization
    max_num_frames: int = GLOBAL_INIT.max_num_frames
    torch_dtype: str = "auto"
    device: str = "cuda"
    use_fast: bool = GLOBAL_INIT.use_fast
    target_fps: int = GLOBAL_INIT.target_fps


class LLaVAOneVisionvLLMSamplingConfig(ModelSamplingConfig):
    temperature: float = GLOBAL_SAMPLING.temperature
    top_p: float = GLOBAL_SAMPLING.top_p
    top_k: int = GLOBAL_SAMPLING.top_k
    max_tokens: int = GLOBAL_SAMPLING.max_tokens
    # use_beam_search: bool = False # beam search is not supported in this version of vLLM, expecially for VLLM
    # best_of: int = GLOBAL_SAMPLING.best_of # use best_of is used toget ther with the beam search, being treated as the beam width. "n must be 1 when using greedy sampling"


class LLaVAOneVisionvLLMModelConfig(ModelConfig):
    init_config: LLaVAOneVisionvLLMInitConfig = Field(
        default_factory=LLaVAOneVisionvLLMInitConfig
    )
    sampling_config: LLaVAOneVisionvLLMSamplingConfig = Field(
        default_factory=LLaVAOneVisionvLLMSamplingConfig
    )
