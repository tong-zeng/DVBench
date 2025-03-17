from typing import Optional
from pydantic import Field
from dvbench.inference.models.base import (
    ModelInitConfig,
    ModelSamplingConfig,
    ModelConfig,
)


class Qwen2VLHFInitConfig(ModelInitConfig):
    pretrained_model_name_or_path: str = "Qwen/Qwen2-VL-7B-Instruct"
    trust_remote_code: bool = True
    max_model_len: int = 2048
    max_num_seqs: int = 10
    gpu_memory_utilization: float = 0.7
    quantization: Optional[str] = None
    torch_dtype: str = "bfloat16"
    device: str = "cuda"
    use_fast: bool = False


class Qwen2VLHFSamplingConfig(ModelSamplingConfig):
    temperature: float = Field(default=0.7, description="Temperature")
    top_p: float = Field(default=0.9, description="Top-p")
    top_k: int = Field(default=50, description="Top-k")
    max_tokens: int = Field(default=1024, description="Maximum number of tokens")
    repetition_penalty: float = Field(default=1.0, description="Repetition penalty")
    stop_token_ids: list = Field(default_factory=list)


class Qwen2VLHFModelConfig(ModelConfig):
    init_config: Qwen2VLHFInitConfig = Field(default_factory=Qwen2VLHFInitConfig)
    sampling_config: Qwen2VLHFSamplingConfig = Field(
        default_factory=Qwen2VLHFSamplingConfig
    )
