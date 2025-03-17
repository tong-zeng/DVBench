from typing import Any
from pydantic import Field, ConfigDict
import torch
from driving_vllm_arena.inference.models.base import (
    ModelInitConfig,
    ModelSamplingConfig,
    ModelConfig,
)
from pathlib import Path

from driving_vllm_arena.inference.configs.app import GLOBAL_SAMPLING, GLOBAL_INIT


class VideoChatGptHFInitConfig(ModelInitConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    local_repo_path: str = str((Path.home() / "PyPackages" / "Video-ChatGPT").resolve())
    pretrained_model_name_or_path: str = "MBZUAI/Video-ChatGPT-7B"
    max_frames: int = GLOBAL_INIT.max_num_frames
    target_fps: int = GLOBAL_INIT.target_fps
    device: str = "cuda"
    torch_dtype: torch.dtype = GLOBAL_INIT.torch_dtype


class VideoChatGptHFSamplingConfig(ModelSamplingConfig):
    temperature: float = GLOBAL_SAMPLING.temperature
    do_sample: bool = GLOBAL_SAMPLING.do_sample
    num_beams: int = GLOBAL_SAMPLING.num_beams
    max_new_tokens: int = GLOBAL_SAMPLING.max_new_tokens
    top_p: float = GLOBAL_SAMPLING.top_p


class VideoChatGptHFConfig(ModelConfig):
    init_config: VideoChatGptHFInitConfig = Field(
        default_factory=VideoChatGptHFInitConfig
    )
    sampling_config: VideoChatGptHFSamplingConfig = Field(
        default_factory=VideoChatGptHFSamplingConfig
    )
