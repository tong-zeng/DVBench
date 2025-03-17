from typing import Optional, Any
from pydantic import BaseModel, Field, ConfigDict
from pathlib import Path
import torch

TARGET_FPS = 2
TARGET_FRAMES_NUM = TARGET_FPS * 5

MAX_FRAMES = 50

"""Global initialization configuration that defines default values and types"""


class GlobalInitConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    max_model_len: int = Field(
        default=4096, description="The maximum sequence length for the model"
    )
    trust_remote_code: bool = Field(
        default=True, description="Whether to trust remote code"
    )
    gpu_memory_utilization: float = Field(
        default=0.95, description="The GPU memory utilization target"
    )
    quantization: Optional[str] = Field(
        default=None, description="The quantization level"
    )
    target_fps: int = Field(
        default=TARGET_FPS, description="The target FPS for video processing"
    )
    max_num_frames: int = Field(
        default=MAX_FRAMES, description="Maximum number of frames to process"
    )
    use_fast: bool = Field(
        default=False, description="Whether to use fast decoding mode"
    )
    torch_dtype: Any = Field(default=torch.float16, description="The torch dtype")
    device: str = Field(default="cuda", description="The device to use")


class GlobalSamplingConfig(BaseModel):
    """Global sampling configuration that defines default values and types"""

    temperature: float = Field(
        default=0.2, description="Temperature for sampling (0.0 means deterministic)"
    )
    top_p: float = Field(default=0.6, description="Top-p sampling threshold")
    top_k: int = Field(default=50, description="Top-k sampling threshold")
    do_sample: bool = Field(default=True, description="HF: Whether to use sampling")
    num_beams: int = Field(
        default=1,
        description="HF: Number of beams for beam search. 1 means no beam search.",
    )
    max_new_tokens: int = Field(
        default=2048, description="HF: Maximum new tokens to generate"
    )

    max_tokens: int = Field(
        default=2048, description="vLLM: Maximum tokens to generate"
    )
    use_beam_search: bool = Field(
        default=False, description="vLLM <=0.6: Whether to use beam search"
    )
    best_of: int = Field(
        default=3, description="vLLM: Beam width when beam search is enabled"
    )
    repetition_penalty: float = Field(default=1.0, description="Repetition penalty")


PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


# Create global config instances
GLOBAL_INIT = GlobalInitConfig()
GLOBAL_SAMPLING = GlobalSamplingConfig()

SYSTEM_PROMPT = """As an AI assistant for video analysis, your role is to examine footage from the front dash camera of an autonomous vehicle and accurately answer multiple-choice questions based on the video.

You will be provided with {num_frames} separate frames uniformly sampled from the video, the frames are provided in chronological order of the video. Please analyze these images and provide the answer to the following question about the video content.

Instructions:
1. Carefully analyze the objects, environment, and events within the video to understand its context and details.
2. Select the correct answer from the provided options by identifying only the option letter (e.g., A, B, C, or D).
3. Respond solely with the correct option letter, ensuring precision in your choice based on the observed content.

Based on your observations, select the best option that accurately addresses the question."""
