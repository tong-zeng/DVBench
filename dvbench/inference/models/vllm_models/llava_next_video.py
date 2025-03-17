from typing import List, Optional, Any, Union
from vllm import PromptType
from vllm.assets.video import VideoAsset
from transformers import ProcessorMixin
from dvbench.inference.models.vllm_models.base_vllm_model import (
    BaseVLLMModel,
)
from dvbench.inference.models.base import ModelInfo
from dvbench.inference.models.vllm_models.vllm_model_configs.config_llava_next_video import (
    LLaVANextVideovLLMInitConfig,
    LLaVANextVideovLLMSamplingConfig,
    LLaVANextVideovLLMModelConfig,
)
from dvbench.inference.models.model_test import test_model


class LLaVANextVideovLLM(
    BaseVLLMModel[
        LLaVANextVideovLLMInitConfig,
        LLaVANextVideovLLMSamplingConfig,
        LLaVANextVideovLLMModelConfig,
        ProcessorMixin,
    ]
):
    def __init__(
        self,
        configs: Union[
            LLaVANextVideovLLMModelConfig, str
        ] = LLaVANextVideovLLMModelConfig(),
        **kwargs,
    ):
        super().__init__(configs, **kwargs)

        if self.processor is None:
            raise ValueError(
                "Processor is required for LLaVANextVideoModel but failed to load"
            )

    @staticmethod
    def get_info() -> ModelInfo:
        return ModelInfo(
            id="llava-hf/LLaVA-NeXT-Video-7B-hf",
            modality="video",
            short_model_size="7B",
            full_model_size=None,
            short_name="LLaVA-NeXT-Video",
            long_name="LLaVA-NeXT-Video 7B",
            link="https://huggingface.co/llava-hf/LLaVA-NeXT-Video-7B-hf",
            description="LLaVA-NeXT-Video is a multimodal model that can process both images and videos",
        )

    def prepare_inputs(
        self,
        prompts: Union[str, List[str]],
        videos: Optional[List[List[VideoAsset]]] = None,
        target_fps: int = -1,
    ) -> List[PromptType]:
        """
        Prepare inputs for the model.
        Args:
            prompts: Single prompt or list of prompts
            videos: List of video lists (one list of videos per prompt)
        """

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
                    and all(isinstance(v, VideoAsset) for v in sublist)
                    for sublist in videos
                )
            ):
                raise ValueError("videos must be List[List[VideoAsset]]")

            # Check length match
            if len(prompts) != len(videos):
                raise ValueError(
                    f"Number of prompts ({len(prompts)}) must match number of video lists ({len(videos)})"
                )

        # Prepare all inputs
        all_inputs = []
        for idx, prompt in enumerate(prompts):
            # Create conversation format
            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video"},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Apply chat template
            prompt_text = self.processor.apply_chat_template(  # type: ignore
                conversation, add_generation_prompt=True, tokenize=False
            )

            # If no videos provided, just use the prompt
            if not videos or idx >= len(videos):
                all_inputs.append({"prompt": prompt_text})
                continue

            # Prepare inputs with videos
            video_list = videos[idx]
            if not video_list:
                all_inputs.append({"prompt": prompt_text})
                continue

            all_inputs.append(
                {
                    "prompt": prompt_text,
                    "multi_modal_data": {
                        "video": video_list[
                            0
                        ].np_ndarrays  # Take first video if multiple provided
                    },
                }
            )

        return all_inputs

    def generate(
        self,
        prompts: Union[str, List[str]],
        videos: Optional[List[List[VideoAsset]]] = None,
        target_fps: int = -1,
        **kwargs,
    ) -> List[Any]:
        """
        Generate outputs for given prompts and videos.
        Args:
            prompts: Single prompt or list of prompts
            videos: List of video lists (one list of videos per prompt)
        """
        inputs = self.prepare_inputs(prompts, videos, target_fps)
        sampling_params = self.get_sampling_params()
        outputs = self.llm.generate(inputs, sampling_params=sampling_params)
        return outputs


if __name__ == "__main__":
    # model_name = "llava-hf/LLaVA-NeXT-Video-7B-hf"
    # Initialize model
    model = LLaVANextVideovLLM()

    # Test the model
    test_model(model)
