from typing import List, Optional, Any, Union
from transformers import PreTrainedModel
from transformers.models.video_llava.processing_video_llava import VideoLlavaProcessor
from transformers.models.video_llava.modeling_video_llava import (
    VideoLlavaForConditionalGeneration,
)
from dvbench.inference.models.base import ModelInfo
from dvbench.inference.models.hf_models.base_hf_model import BaseHFModel
from dvbench.inference.models.hf_models.hf_model_configs.config_video_llava import (
    VideoLLaVAHFInitConfig,
    VideoLLaVAHFSamplingConfig,
    VideoLLaVAHFConfig,
)
from dvbench.inference.ultils.video_processing import (
    pyav_extract_frames_by_fps,
)
from dvbench.inference.models.model_test import test_model
from dvbench.inference.configs.app import TARGET_FPS


class VideoLLaVAHF(
    BaseHFModel[
        VideoLLaVAHFInitConfig,
        VideoLLaVAHFSamplingConfig,
        VideoLLaVAHFConfig,
        VideoLlavaProcessor,
    ]
):
    processor: VideoLlavaProcessor
    model: VideoLlavaForConditionalGeneration

    def __init__(
        self, configs: VideoLLaVAHFConfig | str = VideoLLaVAHFConfig(), **kwargs
    ):
        super().__init__(configs, **kwargs)

    @staticmethod
    def get_info() -> ModelInfo:
        return ModelInfo(
            id="LanguageBind/Video-LLaVA-7B-hf",
            modality="video",
            short_model_size="7B",
            full_model_size=None,
            short_name="Video-LLaVA",
            long_name="Video-LLaVA 7B",
            link="https://huggingface.co/LanguageBind/Video-LLaVA-7B-hf",
            description="Video-LLaVA is a video-language model for video understanding and generation",
        )

    @staticmethod
    def _load_processor_static(init_config) -> Optional[VideoLlavaProcessor]:
        try:
            processor_output = VideoLlavaProcessor.from_pretrained(
                init_config.pretrained_model_name_or_path,
                trust_remote_code=init_config.trust_remote_code,
            )
            # from_pretrained might return either a processor or a tuple containing the processor and additional data
            return (
                processor_output[0]
                if isinstance(processor_output, tuple)
                else processor_output
            )
        except (OSError, ValueError) as e:
            print(f"Error loading processor: {e}")
            return None

    @staticmethod
    def _initialize_model_static(
        init_config: VideoLLaVAHFInitConfig,
    ) -> PreTrainedModel:
        if not hasattr(init_config, "torch_dtype"):
            init_config.torch_dtype = "float16"

        return VideoLlavaForConditionalGeneration.from_pretrained(
            init_config.pretrained_model_name_or_path,
            trust_remote_code=init_config.trust_remote_code,
            torch_dtype=init_config.torch_dtype,
        )

    def prepare_inputs(
        self,
        prompts: Union[str, List[str]],
        videos: Optional[List[List[str]]] = None,
        target_fps: int = TARGET_FPS,
    ):
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

        # Process each video and prompt pair
        # Create empty video lists if videos is None
        videos = videos or [[] for _ in prompts]
        prompt_list = []
        clip_list = []
        for idx, (prompt, video_list) in enumerate(zip(prompts, videos)):
            clip = pyav_extract_frames_by_fps(
                video_list[0], target_fps
            )  # The model only supports single video input
            clip_list.append(clip)
            prompt_list.append(f"USER: <video>{prompt} ASSISTANT:")

        # https://huggingface.co/docs/transformers/main/en/model_doc/video_llava
        # We advise users to use padding_side=“left” when computing batched generation as it leads to more accurate results.
        # Simply make sure to call processor.tokenizer.padding_side = “left” before generating.
        self.processor.tokenizer.padding_side = "left"  # type: ignore
        inputs = self.processor(
            text=prompt_list, videos=clip_list, padding=True, return_tensors="pt"
        ).to(self.model.device)

        return inputs

    def generate(
        self,
        prompts: Union[str, List[str]],
        videos: Optional[List[List[str]]] = None,
        target_fps: int = TARGET_FPS,
        **kwargs,
    ) -> List[Any]:
        inputs = self.prepare_inputs(prompts, videos, target_fps)

        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.sampling_config.max_tokens,
            temperature=self.sampling_config.temperature,
            top_k=self.sampling_config.top_k,
            top_p=self.sampling_config.top_p,
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        # generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        return response


if __name__ == "__main__":
    model = VideoLLaVAHF()
    test_model(model)
