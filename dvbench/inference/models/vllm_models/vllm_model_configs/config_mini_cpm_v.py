from typing import Optional
from pydantic import Field
from dvbench.inference.models.base import (
    ModelInitConfig,
    ModelSamplingConfig,
    ModelConfig,
)
from dvbench.inference.configs.app import GLOBAL_INIT, GLOBAL_SAMPLING


class MiniCPMVvLLMInitConfig(ModelInitConfig):
    # Model-specific settings
    pretrained_model_name_or_path: str = "openbmb/MiniCPM-V-2_6"

    # Settings following global configuration
    max_model_len: int = GLOBAL_INIT.max_model_len
    trust_remote_code: bool = GLOBAL_INIT.trust_remote_code
    gpu_memory_utilization: float = GLOBAL_INIT.gpu_memory_utilization
    quantization: Optional[str] = GLOBAL_INIT.quantization
    target_fps: int = GLOBAL_INIT.target_fps
    max_num_frames: int = GLOBAL_INIT.max_num_frames
    use_fast: bool = GLOBAL_INIT.use_fast
    device: str = "cuda"


class MiniCPMVvLLMSamplingConfig(ModelSamplingConfig):
    # All settings following global configuration
    temperature: float = GLOBAL_SAMPLING.temperature
    top_p: float = GLOBAL_SAMPLING.top_p
    max_tokens: int = GLOBAL_SAMPLING.max_tokens
    repetition_penalty: float = Field(default=1.0, description="Repetition penalty")
    stop_token_ids: list = Field(default_factory=list)
    use_beam_search: bool = GLOBAL_SAMPLING.use_beam_search
    best_of: int = GLOBAL_SAMPLING.best_of


class MiniCPMVvLLMModelConfig(ModelConfig):
    init_config: MiniCPMVvLLMInitConfig = Field(default_factory=MiniCPMVvLLMInitConfig)
    sampling_config: MiniCPMVvLLMSamplingConfig = Field(
        default_factory=MiniCPMVvLLMSamplingConfig
    )
