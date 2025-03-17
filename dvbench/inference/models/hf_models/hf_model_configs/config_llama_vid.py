from typing import Optional
from pydantic import Field, ConfigDict
import torch
from dvbench.inference.models.base import (
    ModelInitConfig,
    ModelSamplingConfig,
    ModelConfig,
)

from dvbench.inference.configs.app import GLOBAL_SAMPLING, GLOBAL_INIT


class LLaMAVidHFInitConfig(ModelInitConfig):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    pretrained_model_name_or_path: str = "YanweiLi/llama-vid-7b-full-224-video-fps-1"
    trust_remote_code: bool = GLOBAL_INIT.trust_remote_code
    max_num_frames: int = GLOBAL_INIT.max_num_frames
    target_fps: int = GLOBAL_INIT.target_fps
    device: str = "cuda"
    torch_dtype: torch.dtype = GLOBAL_INIT.torch_dtype
    use_fast: bool = GLOBAL_INIT.use_fast
    conv_mode: str = Field(default="vicuna_v1", description="Conversation mode")


class LLaMAVidHFSamplingConfig(ModelSamplingConfig):
    temperature: float = GLOBAL_SAMPLING.temperature
    top_p: Optional[float] = GLOBAL_SAMPLING.top_p
    do_sample: bool = GLOBAL_SAMPLING.do_sample
    num_beams: int = GLOBAL_SAMPLING.num_beams
    max_new_tokens: int = GLOBAL_SAMPLING.max_new_tokens
    return_dict_in_generate: bool = Field(
        default=True, description="Return dict in generate"
    )
    output_scores: bool = Field(default=True, description="Output scores")
    use_cache: bool = Field(default=True, description="Use cache")


class LLaMAVidHFConfig(ModelConfig):
    init_config: LLaMAVidHFInitConfig = Field(default_factory=LLaMAVidHFInitConfig)
    sampling_config: LLaMAVidHFSamplingConfig = Field(
        default_factory=LLaMAVidHFSamplingConfig
    )
