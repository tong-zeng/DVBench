from typing import List, Optional, Any, Union
from pydantic import ConfigDict
from transformers import (
    PreTrainedModel,
    AutoModelForCausalLM,
    AutoProcessor,
    AutoImageProcessor,
    ProcessorMixin,
)
import torch
import os
import numpy as np
from PIL import Image
from decord import VideoReader, cpu  # type: ignore

from ChatUniVi.constants import *
from ChatUniVi.conversation import conv_templates, SeparatorStyle
from ChatUniVi.model.builder import load_pretrained_model
from ChatUniVi.utils import disable_torch_init
from ChatUniVi.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from ChatUniVi import ChatUniViLlamaForCausalLM

from dvbench.inference.models.base import ModelInfo
from dvbench.inference.models.hf_models.base_hf_model import BaseHFModel
from dvbench.inference.models.hf_models.hf_model_configs.config_chat_uni_vi import (
    ChatUniViHFInitConfig,
    ChatUniViHFSamplingConfig,
    ChatUniViHFConfig,
)
from dvbench.inference.models.model_test import test_model
from dvbench.inference.configs.app import MAX_FRAMES, TARGET_FPS


def _get_rawvideo_dec(
    video_path,
    image_processor,
    max_frames=MAX_FRAMES,
    image_resolution=224,
    video_framerate=1,
    s=None,
    e=None,
):
    # speed up video decode via decord.

    if s is None:
        start_time, end_time = None, None
    else:
        start_time = int(s)
        end_time = int(e)  # type: ignore
        start_time = start_time if start_time >= 0.0 else 0.0
        end_time = end_time if end_time >= 0.0 else 0.0
        if start_time > end_time:
            start_time, end_time = end_time, start_time
        elif start_time == end_time:
            end_time = start_time + 1

    if os.path.exists(video_path):
        vreader = VideoReader(video_path, ctx=cpu(0))
    else:
        print(video_path)
        raise FileNotFoundError

    fps = vreader.get_avg_fps()
    f_start = 0 if start_time is None else int(start_time * fps)
    f_end = int(
        min(1000000000 if end_time is None else end_time * fps, len(vreader) - 1)
    )
    num_frames = f_end - f_start + 1
    if num_frames > 0:
        # T x 3 x H x W
        sample_fps = int(video_framerate)
        t_stride = int(round(float(fps) / sample_fps))

        all_pos = list(range(f_start, f_end + 1, t_stride))
        if len(all_pos) > max_frames:
            sample_pos = [
                all_pos[_]
                for _ in np.linspace(0, len(all_pos) - 1, num=max_frames, dtype=int)
            ]
        else:
            sample_pos = all_pos

        patch_images = [
            Image.fromarray(f) for f in vreader.get_batch(sample_pos).asnumpy()
        ]

        patch_images = torch.stack(
            [
                image_processor.preprocess(img, return_tensors="pt")["pixel_values"][0]
                for img in patch_images
            ]
        )
        slice_len = patch_images.shape[0]

        return patch_images, slice_len
    else:
        print("video path: {} error.".format(video_path))


class ChatUniViHF(
    BaseHFModel[
        ChatUniViHFInitConfig,
        ChatUniViHFSamplingConfig,
        ChatUniViHFConfig,
        ProcessorMixin,
    ]
):
    processor: AutoProcessor | None = None
    model: ChatUniViLlamaForCausalLM
    image_processor: Any | None = None

    def __init__(
        self, configs: ChatUniViHFConfig | str = ChatUniViHFConfig(), **kwargs
    ):
        self.processor = None
        self.image_processor = None

        # Call parent constructor
        super().__init__(configs, **kwargs)

        disable_torch_init()

        # Handle special tokens
        self._setup_special_tokens()
        self._setup_vision_tower()
        self._setup_context_len()

        # Move model to specified device
        for n, m in self.model.named_modules():
            m.to(device=self.init_config.device, dtype=torch.float16)  # type: ignore

    def _setup_context_len(self):
        if hasattr(self.model.config, "max_sequence_length"):
            self.context_len = self.model.config.max_sequence_length
        else:
            self.context_len = 2048

    def _setup_special_tokens(self):
        """Setup special tokens for the model"""
        mm_use_im_start_end = getattr(self.model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(
            self.model.config, "mm_use_im_patch_token", True
        )

        if mm_use_im_patch_token:
            self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            self.tokenizer.add_tokens(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
            )
        self.model.resize_token_embeddings(len(self.tokenizer))

    def _setup_vision_tower(self):
        """Setup and load the vision tower"""
        vision_tower = self.model.get_vision_tower()
        if not vision_tower.is_loaded:
            vision_tower.load_model()
            # Move vision tower to same device as model
            vision_tower.to(device=self.init_config.device)
        self.image_processor = vision_tower.image_processor

    @staticmethod
    def get_info() -> ModelInfo:
        return ModelInfo(
            id="Chat-UniVi/Chat-UniVi",
            modality="video",
            short_model_size="7B",
            full_model_size=None,
            short_name="Chat-UniVi",
            long_name="Chat-UniVi 7B",
            link="https://huggingface.co/Chat-UniVi/Chat-UniVi",
            description="Chat-UniVi is a video-language model for video understanding and conversation",
        )

    @staticmethod
    def _load_processor_static(init_config) -> Optional[AutoProcessor]:
        return None

    @staticmethod
    def _initialize_model_static(init_config: ChatUniViHFInitConfig) -> PreTrainedModel:
        model = AutoModelForCausalLM.from_pretrained(
            init_config.pretrained_model_name_or_path,
            trust_remote_code=init_config.trust_remote_code,
            torch_dtype=init_config.torch_dtype,
        )
        # Move model to specified device after initialization
        model = model.to(init_config.device)

        return model

    def prepare_inputs(
        self,
        prompts: Union[str, List[str]],
        videos: Optional[List[List[str]]] = None,
        target_fps: int = -1,
    ):
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

        videos = videos or [[] for _ in prompts]
        input_ids_list = []
        video_frames_list = []
        for prompt, video_list in zip(prompts, videos):
            # Only supports single video input
            patch_images_tuple = _get_rawvideo_dec(
                video_list[0],
                self.image_processor,
                max_frames=MAX_FRAMES,
                video_framerate=target_fps,
            )
            if patch_images_tuple is None:
                raise ValueError(f"Failed to load video {video_list[0]}")

            video_frames, slice_len = patch_images_tuple

            # Prepare prompt
            if self.model.config.mm_use_im_start_end:
                qs = (
                    DEFAULT_IM_START_TOKEN
                    + DEFAULT_IMAGE_TOKEN * slice_len
                    + DEFAULT_IM_END_TOKEN
                    + "\n"
                    + prompt
                )
            else:
                qs = DEFAULT_IMAGE_TOKEN * slice_len + "\n" + prompt

            # Setup conversation
            conv = conv_templates[self.sampling_config.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            # Prepare inputs
            input_ids = (
                tokenizer_image_token(
                    prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)  # type: ignore
                .cuda()
            )  # type: ignore

            input_ids_list.append(input_ids)
            video_frames_list.append(video_frames)

        return input_ids_list, video_frames_list

    def generate(
        self,
        prompts: Union[str, List[str]],
        videos: Optional[List[List[str]]] = None,
        target_fps: int = TARGET_FPS,
        **kwargs,
    ) -> List[str]:
        input_ids_list, video_frames_list = self.prepare_inputs(
            prompts, videos, target_fps
        )
        outputs_list = []

        # Process each input separately to maintain proper image-text alignment
        for batch_idx, (input_ids, video_frames) in enumerate(
            zip(input_ids_list, video_frames_list)
        ):
            # Ensure input_ids is 2D and video_frames is properly batched
            input_ids = input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids
            video_frames = (
                video_frames.unsqueeze(0) if video_frames.dim() == 3 else video_frames
            )

            conv = conv_templates[self.init_config.conv_mode].copy()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]

            stopping_criteria = KeywordsStoppingCriteria(
                keywords, self.tokenizer, input_ids
            )

            # Move input_ids to the same device as the model
            input_ids = input_ids.to(self.init_config.device)
            video_frames = video_frames.to(self.init_config.device)

            output = self.model.generate(
                input_ids,
                images=video_frames,  # video_frames.half().cuda(), #[item.half().cuda() for item in video_frames_list], # type: ignore
                do_sample=self.sampling_config.do_sample,
                temperature=self.sampling_config.temperature,
                top_p=self.sampling_config.top_p,
                num_beams=self.sampling_config.num_beams,
                max_new_tokens=self.sampling_config.max_new_tokens,
                use_cache=self.sampling_config.use_cache,
                stopping_criteria=[stopping_criteria],  # type: ignore
                output_scores=self.sampling_config.output_scores,
                return_dict_in_generate=self.sampling_config.return_dict_in_generate,
            )

            # Process output
            output_ids = output.sequences  # type: ignore
            input_token_len = input_ids.shape[1]
            outputs = self.tokenizer.batch_decode(
                output_ids[:, input_token_len:], skip_special_tokens=True
            )[0].strip()

            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]
            outputs_list.append(outputs.strip())

        return outputs_list


if __name__ == "__main__":
    model = ChatUniViHF()
    test_model(model)
