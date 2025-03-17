from typing import Optional
from pydantic import Field
from driving_vllm_arena.inference.models.base import (
    ModelInitConfig,
    ModelSamplingConfig,
    ModelConfig,
)

from driving_vllm_arena.inference.configs.app import GLOBAL_SAMPLING, GLOBAL_INIT


class ChatUniViHFInitConfig(ModelInitConfig):
    pretrained_model_name_or_path: str = "Chat-UniVi/Chat-UniVi-13B"
    trust_remote_code: bool = GLOBAL_INIT.trust_remote_code
    max_num_frames: int = GLOBAL_INIT.max_num_frames
    target_fps: int = GLOBAL_INIT.target_fps
    device: str = "cuda"
    torch_dtype: str = "auto"
    use_fast: bool = GLOBAL_INIT.use_fast


class ChatUniViHFSamplingConfig(ModelSamplingConfig):
    temperature: float = GLOBAL_SAMPLING.temperature
    top_p: Optional[float] = GLOBAL_SAMPLING.top_p
    do_sample: bool = GLOBAL_SAMPLING.do_sample
    num_beams: int = GLOBAL_SAMPLING.num_beams
    max_new_tokens: int = GLOBAL_SAMPLING.max_new_tokens
    conv_mode: str = Field(default="simple", description="The conversation mode")
    return_dict_in_generate: bool = Field(
        default=True, description="Return dict in generate"
    )
    output_scores: bool = Field(default=True, description="Output scores")
    use_cache: bool = Field(default=True, description="Use cache")


class ChatUniViHFConfig(ModelConfig):
    init_config: ChatUniViHFInitConfig = Field(default_factory=ChatUniViHFInitConfig)
    sampling_config: ChatUniViHFSamplingConfig = Field(
        default_factory=ChatUniViHFSamplingConfig
    )
