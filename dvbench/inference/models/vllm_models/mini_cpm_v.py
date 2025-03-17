from typing import List, Optional, Any, Union
from PIL import Image
from pydantic import ConfigDict
from transformers import ProcessorMixin

from dvbench.inference.models.vllm_models.base_vllm_model import (
    BaseVLLMModel,
)
from dvbench.inference.models.base import ModelInfo
from dvbench.inference.models.vllm_models.vllm_model_configs.config_mini_cpm_v import (
    MiniCPMVvLLMInitConfig,
    MiniCPMVvLLMSamplingConfig,
    MiniCPMVvLLMModelConfig,
)

from dvbench.inference.models.model_test import test_model

from vllm import LLM, SamplingParams

from decord import VideoReader, cpu

from dvbench.inference.configs.app import SYSTEM_PROMPT

from typing import cast


class MiniCPMVvLLM(
    BaseVLLMModel[
        MiniCPMVvLLMInitConfig,
        MiniCPMVvLLMSamplingConfig,
        MiniCPMVvLLMModelConfig,
        ProcessorMixin,
    ]
):
    """MiniCPM-V model implementation."""

    def __init__(
        self,
        configs: Union[MiniCPMVvLLMModelConfig, str] = MiniCPMVvLLMModelConfig(),
        **kwargs,
    ):
        """Initialize MiniCPM-V model."""
        super().__init__(configs, **kwargs)

        self._initialize_stop_tokens()

    def _initialize_stop_tokens(self) -> None:
        """Initialize model-specific stop tokens."""
        stop_tokens = ["<|im_end|>", "<|endoftext|>"]
        self.sampling_config.stop_token_ids = [
            self.tokenizer.convert_tokens_to_ids(i) for i in stop_tokens
        ]

    @staticmethod
    def get_info() -> ModelInfo:
        return ModelInfo(
            id="openbmb/MiniCPM-V-2_6",
            modality="video",
            short_model_size="2.6B",
            full_model_size=None,
            short_name="MiniCPM-V",
            long_name="MiniCPM-V 2.6B",
            link="https://huggingface.co/openbmb/MiniCPM-V-2_6",
            description="MiniCPM-V is a multimodal model that can process both images and videos",
        )

    def get_sampling_params(self) -> SamplingParams:
        return SamplingParams(**self.sampling_config.model_dump())

    @staticmethod
    def _initialize_llm_static(init_config) -> LLM:
        # return  LLM(
        #     model=init_config.pretrained_model_name_or_path,
        #     # trust_remote_code=init_config.trust_remote_code,
        #     # gpu_memory_utilization=init_config.gpu_memory_utilization,
        #     # max_model_len=init_config.max_model_len
        #     trust_remote_code=True,
        #     gpu_memory_utilization=0.85,
        #     max_model_len=4096
        # )
        return LLM(
            model=init_config.pretrained_model_name_or_path,
            trust_remote_code=init_config.trust_remote_code,
            max_model_len=init_config.max_model_len,
            gpu_memory_utilization=init_config.gpu_memory_utilization,
            quantization=init_config.quantization,
        )

    def _encode_video(self, filepath: str) -> List[Image.Image]:
        def uniform_sampling(l, n):
            gap = len(l) / n
            idxs = [int(i * gap + gap / 2) for i in range(n)]
            return [l[i] for i in idxs]

        vr = VideoReader(filepath, ctx=cpu(0))
        sample_fps = round(vr.get_avg_fps() / self.init_config.target_fps)
        frame_idx = [i for i in range(0, len(vr), sample_fps)]

        if len(frame_idx) > self.init_config.max_num_frames:
            frame_idx = uniform_sampling(frame_idx, self.init_config.max_num_frames)

        video_frames = vr.get_batch(frame_idx).asnumpy()
        return [Image.fromarray(frame.astype("uint8")) for frame in video_frames]

    def prepare_inputs(
        self,
        prompts: Union[str, List[str]],
        videos: Optional[List[List[str]]] = None,
        target_fps: int = -1,
    ) -> List[Any]:
        if target_fps == -1:
            target_fps = self.init_config.target_fps

        if isinstance(prompts, str):
            prompts = [prompts]

        if videos is not None:
            if not isinstance(videos, list) or not all(
                isinstance(v, list) for v in videos
            ):
                raise ValueError("videos must be List[List[str]]")
            if len(prompts) != len(videos):
                raise ValueError("Number of prompts must match number of video lists")

        all_inputs = []
        videos = videos or [[] for _ in prompts]

        for prompt, video_list in zip(prompts, videos):
            if not video_list:
                all_inputs.append({"prompt": prompt})
                continue

            video_frames = self._encode_video(video_list[0])

            system_prompt = SYSTEM_PROMPT.format(num_frames=len(video_frames))
            # print(f"System prompt: {system_prompt}")
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": "".join(["(<image>./</image>)"] * len(video_frames))
                    + f"\n{prompt}",
                },
            ]

            # print(f"Messages: {messages}")

            prompt_text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            all_inputs.append(
                {
                    "prompt": prompt_text,
                    "multi_modal_data": {
                        "image": {
                            "images": video_frames,
                            "use_image_id": False,
                            "max_slice_nums": 1 if len(video_frames) > 16 else 2,
                        }
                    },
                }
            )

        return all_inputs

    def generate(
        self,
        prompts: Union[str, List[str]],
        videos: Optional[List[List[str]]] = None,
        target_fps: int = -1,
        **kwargs,
    ) -> List[Any]:
        inputs = self.prepare_inputs(prompts, videos, target_fps)
        sampling_params = self.get_sampling_params()
        return self.llm.generate(inputs, sampling_params=sampling_params)


if __name__ == "__main__":
    configs = MiniCPMVvLLMModelConfig()
    # Add debug prints
    # print("\nDebug direct instantiation:")
    # print(f"Configs type: {type(configs)}")
    # print(f"Configs: {configs}")
    # print(f"Configs dict: {configs.__dict__}")

    model = MiniCPMVvLLM(configs)
    test_model(model)
    # test_model_with_questions(model)
