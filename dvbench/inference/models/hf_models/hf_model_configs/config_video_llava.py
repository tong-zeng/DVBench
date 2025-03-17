from typing import Optional
from pydantic import Field
import torch
from dvbench.inference.models.base import (
    ModelInitConfig,
    ModelSamplingConfig,
    ModelConfig,
)

from dvbench.inference.configs.app import GLOBAL_SAMPLING, GLOBAL_INIT


class VideoLLaVAHFInitConfig(ModelInitConfig):
    pretrained_model_name_or_path: str = "LanguageBind/Video-LLaVA-7B-hf"
    trust_remote_code: bool = GLOBAL_INIT.trust_remote_code
    max_model_len: int = GLOBAL_INIT.max_model_len
    # max_num_seqs: int = GLOBAL_INIT.max_num_seqs
    gpu_memory_utilization: float = GLOBAL_INIT.gpu_memory_utilization
    torch_dtype: torch.dtype = GLOBAL_INIT.torch_dtype
    device: str = GLOBAL_INIT.device
    use_fast: bool = GLOBAL_INIT.use_fast


class VideoLLaVAHFSamplingConfig(ModelSamplingConfig):
    temperature: float = GLOBAL_SAMPLING.temperature
    top_p: float = GLOBAL_SAMPLING.top_p
    top_k: int = GLOBAL_SAMPLING.top_k
    max_tokens: int = GLOBAL_SAMPLING.max_tokens
    repetition_penalty: float = GLOBAL_SAMPLING.repetition_penalty
    stop_token_ids: list = Field(default_factory=list)


class VideoLLaVAHFConfig(ModelConfig):
    init_config: VideoLLaVAHFInitConfig = Field(default_factory=VideoLLaVAHFInitConfig)
    sampling_config: VideoLLaVAHFSamplingConfig = Field(
        default_factory=VideoLLaVAHFSamplingConfig
    )
