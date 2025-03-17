from typing import List, Optional, Any, Union
from vllm.assets.video import VideoAsset
from vllm import PromptType
from transformers import ProcessorMixin
from pydantic import ConfigDict

from dvbench.inference.models.vllm_models.base_vllm_model import (
    BaseVLLMModel,
)
from dvbench.inference.models.base import ModelInfo
from dvbench.inference.models.vllm_models.vllm_model_configs.config_qwen2_vl import (
    Qwen2VLvLLMInitConfig,
    Qwen2VLvLLMSamplingConfig,
    Qwen2VLvLLMModelConfig,
)
from dvbench.inference.models.model_test import test_model
from qwen_vl_utils import process_vision_info
from dvbench.inference.configs.app import TARGET_FPS


class Qwen2VLvLLM(
    BaseVLLMModel[
        Qwen2VLvLLMInitConfig,
        Qwen2VLvLLMSamplingConfig,
        Qwen2VLvLLMModelConfig,
        ProcessorMixin,
    ]
):
    processor: ProcessorMixin

    def __init__(
        self,
        configs: Union[Qwen2VLvLLMModelConfig, str] = Qwen2VLvLLMModelConfig(),
        **kwargs,
    ):
        super().__init__(configs, **kwargs)
        if self.processor is None:
            raise ValueError(
                "Processor is required for Qwen2VLModel but failed to load"
            )

    @staticmethod
    def get_info() -> ModelInfo:
        return ModelInfo(
            id="Qwen/Qwen2-VL-7B-Instruct",
            modality="video",
            short_model_size="7B",
            full_model_size=None,
            short_name="Qwen2-VL",
            long_name="Qwen2 VL 7B",
            link="https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct",
            description="Qwen2 VL is a multimodal model that can process both images and videos",
        )

    def prepare_inputs(
        self,
        prompts: Union[str, List[str]],
        videos: Optional[List[List[str]]] = None,
        target_fps: int = -1,
    ) -> List[PromptType]:
        if target_fps == -1:
            target_fps = self.init_config.target_fps

        # Normalize prompts to List[str]
        if isinstance(prompts, str):
            prompts = [prompts]

        # Validate videos type and structure
        if videos is not None:
            if not (
                isinstance(videos, list)
                and all(
                    isinstance(sublist, list)
                    and all(isinstance(v, str) for v in sublist)
                    for sublist in videos
                )
            ):
                raise ValueError("videos must be List[List[str]]")

            # Check length match
            if len(prompts) != len(videos):
                raise ValueError(
                    f"Number of prompts ({len(prompts)}) must match number of video lists ({len(videos)})"
                )

        all_inputs = []

        # Create empty video lists if videos is None
        videos = videos or [[] for _ in prompts]
        for prompt, video_list in zip(prompts, videos):
            if not video_list:
                all_inputs.append({"prompt": prompt})
                continue
            # Use pil_images property to get the frames as PIL Images

            # The process_vision_info only supports single video input
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": video_list[0], "fps": target_fps},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            prompt_text = self.processor.apply_chat_template(
                conversation, add_generation_prompt=True, tokenize=False
            )

            image_inputs, video_inputs = process_vision_info(conversation)
            mm_data = {}
            if image_inputs is not None:
                mm_data["image"] = image_inputs
            if video_inputs is not None:
                mm_data["video"] = video_inputs

            all_inputs.append(
                {
                    "prompt": prompt_text,
                    "multi_modal_data": mm_data,
                }
            )

        return all_inputs

    def generate(
        self,
        prompts: Union[str, List[str]],
        videos: Optional[List[List[str]]] = None,
        target_fps: int = TARGET_FPS,
        **kwargs,
    ) -> List[Any]:
        """Generate outputs for a single prompt and video."""
        inputs = self.prepare_inputs(prompts, videos, target_fps)
        sampling_params = self.get_sampling_params()
        outputs = self.llm.generate(inputs, sampling_params=sampling_params)
        return outputs


if __name__ == "__main__":
    model = Qwen2VLvLLM()
    test_model(model)
