from typing import Optional
from pydantic import Field, ConfigDict
import torch
from driving_vllm_arena.inference.models.base import (
    ModelInitConfig,
    ModelSamplingConfig,
    ModelConfig,
)
from pathlib import Path

from driving_vllm_arena.inference.configs.app import GLOBAL_SAMPLING, GLOBAL_INIT


class PLLaVAHFInitConfig(ModelInitConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    local_repo_path: str = str((Path.home() / "PyPackages" / "PLLaVA").resolve())
    pretrained_model_name_or_path: str = "ermu2001/pllava-13b"
    max_frames: int = GLOBAL_INIT.max_num_frames
    target_fps: int = GLOBAL_INIT.target_fps
    device: str = "cuda"
    torch_dtype: torch.dtype = GLOBAL_INIT.torch_dtype
    img_resolution: int = Field(
        default=336, description="Resolution"
    )  # DO NOT CHANGE, Specific to PLLaVA
    use_multi_gpus: bool = False


class PLLaVAHFSamplingConfig(ModelSamplingConfig):
    temperature: float = GLOBAL_SAMPLING.temperature
    do_sample: bool = GLOBAL_SAMPLING.do_sample
    num_beams: int = GLOBAL_SAMPLING.num_beams
    max_new_tokens: int = GLOBAL_SAMPLING.max_new_tokens
    min_length: int = Field(
        default=1, description="Minimum length"
    )  # DO NOT CHANGE, Specific to PLLaVA
    top_p: float = GLOBAL_SAMPLING.top_p
    repetition_penalty: float = Field(
        default=1.0, description="Repetition penalty"
    )  # DO NOT CHANGE, Specific to PLLaVA
    length_penalty: float = Field(
        default=1.0, description="Length penalty"
    )  # DO NOT CHANGE, Specific to PLLaVA


class PLLaVAHFConfig(ModelConfig):
    init_config: PLLaVAHFInitConfig = Field(default_factory=PLLaVAHFInitConfig)
    sampling_config: PLLaVAHFSamplingConfig = Field(
        default_factory=PLLaVAHFSamplingConfig
    )
