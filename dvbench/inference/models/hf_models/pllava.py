from typing import List, Optional, Any, Union
from pydantic import ConfigDict
from transformers import (
    PreTrainedModel,
    AutoModelForCausalLM,
    AutoProcessor,
    ProcessorMixin,
    CLIPImageProcessor,
)
import torch
import os
from decord import VideoReader, cpu

from dvbench.inference.models.base import ModelInfo
from dvbench.inference.models.hf_models.base_hf_model import BaseHFModel
from dvbench.inference.models.hf_models.hf_model_configs.config_pllava import (
    PLLaVAHFInitConfig,
    PLLaVAHFSamplingConfig,
    PLLaVAHFConfig,
)
from dvbench.inference.models.model_test import test_model
from dvbench.inference.configs.app import MAX_FRAMES, TARGET_FPS

from huggingface_hub import snapshot_download

import sys
import numpy as np
import logging
import torch
from PIL import Image
import torchvision


def load_video(
    vis_path, max_frames=100, video_framerate=1, resolution=336, s=None, e=None
):
    """
    Load video frames from a video file.

    Parameters:
    vis_path (str): Path to the video file.
    n_clips (int): Number of clips to extract from the video. Defaults to 1.
    num_frm (int): Number of frames to extract from each clip. Defaults to 100.

    Returns:
    list: List of PIL.Image.Image objects representing video frames.
    """
    transforms = torchvision.transforms.Resize(size=resolution)

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

    # Load video with VideoReader
    if os.path.exists(vis_path):
        vr = VideoReader(vis_path, ctx=cpu(0))
    else:
        print(vis_path)
        raise FileNotFoundError

    fps = vr.get_avg_fps()
    f_start = 0 if start_time is None else int(start_time * fps)
    f_end = int(min(1000000000 if end_time is None else end_time * fps, len(vr) - 1))
    num_frames = f_end - f_start + 1
    if num_frames > 0:
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

        # Extract frames as numpy array
        img_array = vr.get_batch(sample_pos).asnumpy()
        # Set target image height and width

        # Convert numpy arrays to PIL Image objects
        clip_imgs = [Image.fromarray(img_array[j]) for j in range(len(sample_pos))]
        clip_imgs = [transforms(img) for img in clip_imgs]

        return clip_imgs

    return None


def get_rank_and_world_size():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return rank, world_size


class PLLaVAHF(
    BaseHFModel[
        PLLaVAHFInitConfig, PLLaVAHFSamplingConfig, PLLaVAHFConfig, ProcessorMixin
    ]
):
    processor: Any | None
    model: PreTrainedModel
    image_processor: CLIPImageProcessor | None

    def __init__(self, configs: PLLaVAHFConfig | str = PLLaVAHFConfig(), **kwargs):
        self.processor = None
        self.image_processor = None

        super().__init__(configs, **kwargs)
        pretrained_model_name_or_path = self.init_config.pretrained_model_name_or_path

        local_repo_path = self.init_config.local_repo_path
        assert local_repo_path is not None, "local_repo_path must be provided"
        sys.path.insert(0, local_repo_path)
        print(f"sys.path: {sys.path}")

        try:
            from tasks.eval.model_utils import load_pllava
            from tasks.eval.model_utils import pllava_answer
            from tasks.eval.eval_utils import (
                conv_templates,
                Conversation,
                MultiModalConvStyle,
            )
        except Exception as err:
            logging.critical(
                "Please first install requirements and set the root path to use PLLaVA. \
                Follow the instructions at https://github.com/magic-research/PLLaVA."
            )
            raise err

        rank, world_size = get_rank_and_world_size()

        self.nframe = 16  # We need to set this really value for the video, we will update this later
        self.use_lora = True
        self.lora_alpha = 4
        self.pooling_shape = (16, 12, 12)
        self.RESOLUTION = self.init_config.img_resolution
        self.model_path = pretrained_model_name_or_path

        # remind that, once the model goes larger (30B+) may cause the memory to be heavily used up. Even Tearing Nodes.
        weight_dir = snapshot_download(self.model_path)
        self.model, self.processor = load_pllava(
            pretrained_model_name_or_path,
            num_frames=self.nframe,
            use_lora=self.use_lora,
            weight_dir=weight_dir,
            lora_alpha=self.lora_alpha,
            pooling_shape=self.pooling_shape,
            use_multi_gpus=self.init_config.use_multi_gpus,
        )

        #  position embedding
        self.model = self.model.to(torch.device(rank))  # type: ignore
        self.model = self.model.eval()

        SYSTEM_MVBENCH = "Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.\n"
        self.conv = Conversation(
            system=SYSTEM_MVBENCH,
            roles=("USER: ", "ASSISTANT:"),
            messages=[],
            sep=[" ", "</s>"],
            mm_token="<image>\n",
            mm_style=MultiModalConvStyle.MM_INTERLEAF,
        )
        self.conv_templates = conv_templates

        self.pllava_answer = pllava_answer

    def _update_model_config_nframe(self, nframe):
        self.nframe = nframe
        self.model.config.num_frames = nframe  # type: ignore
        self.model.multi_modal_projector.num_frames = nframe  # type: ignore

    @staticmethod
    def get_info() -> ModelInfo:
        return ModelInfo(
            id="ermu2001/pllava-13b",
            modality="video",
            short_model_size="13B",
            full_model_size=None,
            short_name="PLLaVA",
            long_name="PLLaVA",
            link="https://huggingface.co/ermu2001/pllava-13b",
            description="PLLaVA is a video-language model",
        )

    @staticmethod
    def _load_tokenizer_static(init_config):
        return None

    @staticmethod
    def _load_processor_static(init_config) -> Optional[AutoProcessor]:
        return None

    @staticmethod
    def _initialize_model_static(
        init_config: PLLaVAHFInitConfig,
    ) -> PreTrainedModel | None:
        return None

    def prepare_inputs(
        self,
        prompts: Union[str, List[str]],
        videos: Optional[List[List[str]]] = None,
        target_fps: int = -1,
    ):
        if target_fps == -1:
            target_fps = self.init_config.target_fps

        max_frames = self.init_config.max_frames

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
        prompt_list = []
        video_frames_list = []
        for prompt, video_list in zip(prompts, videos):
            # Add debug logging
            print(f"Loading video: {video_list[0]}")

            video_frames = load_video(
                video_list[0],
                max_frames=max_frames,
                video_framerate=target_fps,
                resolution=self.RESOLUTION,
                s=None,
                e=None,
            )

            # Validate video frames were loaded
            if video_frames is None or len(video_frames) == 0:
                raise ValueError(f"Failed to load video frames from {video_list[0]}")

            print(f"Loaded {len(video_frames)} frames")

            prompt_list.append(prompt)
            video_frames_list.append(video_frames)

        return prompt_list, video_frames_list

    def generate(
        self,
        prompts: Union[str, List[str]],
        videos: Optional[List[List[str]]] = None,
        target_fps: int = TARGET_FPS,
        **kwargs,
    ) -> List[str]:
        prompt_list, video_frames_list = self.prepare_inputs(
            prompts, videos, target_fps
        )

        outputs_list = []
        for prompt, video_frames in zip(prompt_list, video_frames_list):
            num_frames = len(video_frames)
            self._update_model_config_nframe(num_frames)

            conv = self.conv.copy()
            conv.user_query(prompt, is_mm=True)

            llm_response, conv = self.pllava_answer(
                conv=conv,
                model=self.model,
                processor=self.processor,
                img_list=video_frames,
                do_sample=self.sampling_config.do_sample,
                max_new_tokens=self.sampling_config.max_new_tokens,
                num_beams=self.sampling_config.num_beams,
                min_length=self.sampling_config.min_length,
                top_p=self.sampling_config.top_p,
                repetition_penalty=self.sampling_config.repetition_penalty,
                length_penalty=int(self.sampling_config.length_penalty),
                temperature=self.sampling_config.temperature,
                stop_criteria_keywords=None,
                print_res=False,
            )

            outputs_list.append(llm_response)

        return outputs_list


if __name__ == "__main__":
    model = PLLaVAHF()
    test_model(model)


# if __name__ == "__main__":


#     # import logging
#     import os
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger(__name__)

#     try:
#         # Initialize model
#         logger.info("Initializing PLLaVA model...")
#         model = PLLaVAHF()

#         # Test with a simple video
#         test_video_path = "/home/username/Projects/DrivingVllmBench/videos/5s/96898040.mp4"
#          # Replace with an actual test video path
#         if not os.path.exists(test_video_path):
#             logger.warning(f"Test video not found at {test_video_path}")
#             logger.info("Please provide a valid video path to test video processing")
#         else:
#             # Test single video
#             prompt = "What is happening in this video?"
#             logger.info(f"Testing single video processing with prompt: {prompt}")

#             message = [
#                 {
#                     "type": "text",
#                     "value": prompt
#                 },
#                 {
#                     "type": "video",
#                     "value": test_video_path
#                 }
#             ]

#             outputs = model.generate_inner(prompt, test_video_path)
#             print(f"=================outputs={outputs}=======================")
#             logger.info(f"Model output: {outputs[0]}")

#             # # Test batch processing
#             # prompts = [
#             #     "What is happening in this video?",
#             #     "Describe the main action in this video."
#             # ]
#             # videos = [[test_video_path], [test_video_path]]
#             # logger.info("Testing batch video processing...")
#             # outputs = model.generate(prompts, videos)
#             # for i, output in enumerate(outputs):
#             #     logger.info(f"Output {i+1}: {output}")


#             model = PLLaVAHF()
#             test_model(model)

#         logger.info("Model initialization and basic tests completed successfully!")

#     except Exception as e:
#         logger.error(f"Error during testing: {str(e)}")
#         import traceback
#         logger.error(traceback.format_exc())
