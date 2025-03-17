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
from dvbench.inference.models.hf_models.hf_model_configs.config_video_chatgpt import (
    VideoChatGptHFInitConfig,
    VideoChatGptHFSamplingConfig,
    VideoChatGptHFConfig,
)
from dvbench.inference.models.model_test import test_model
from dvbench.inference.configs.app import MAX_FRAMES, TARGET_FPS

from huggingface_hub import snapshot_download

import sys
import numpy as np
import logging
import torch
from PIL import Image

# Define constants
DEFAULT_VIDEO_TOKEN = "<video>"
DEFAULT_VIDEO_PATCH_TOKEN = "<vid_patch>"
DEFAULT_VID_START_TOKEN = "<vid_start>"
DEFAULT_VID_END_TOKEN = "<vid_end>"


def load_video(vis_path, n_clips=1, max_frames=100, video_framerate=1, s=None, e=None):
    """
    Load video frames from a video file.

    Parameters:
    vis_path (str): Path to the video file.
    n_clips (int): Number of clips to extract from the video. Defaults to 1.
    num_frm (int): Number of frames to extract from each clip. Defaults to 100.

    Returns:
    list: List of PIL.Image.Image objects representing video frames.
    """

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
    # total_frame_num = len(vr)

    # Currently, this function supports only 1 clip
    assert n_clips == 1

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
        target_h, target_w = 224, 224
        # If image shape is not as target, resize it
        if img_array.shape[-3] != target_h or img_array.shape[-2] != target_w:
            img_array = torch.from_numpy(img_array).permute(0, 3, 1, 2).float()
            img_array = torch.nn.functional.interpolate(
                img_array, size=(target_h, target_w)
            )
            img_array = img_array.permute(0, 2, 3, 1).to(torch.uint8).numpy()

        # Reshape array to match number of clips and frames
        img_array = img_array.reshape(
            (
                n_clips,
                len(sample_pos),
                img_array.shape[-3],
                img_array.shape[-2],
                img_array.shape[-1],
            )
        )
        # Convert numpy arrays to PIL Image objects
        clip_imgs = [Image.fromarray(img_array[0, j]) for j in range(len(sample_pos))]

        return clip_imgs


def get_spatio_temporal_features_torch(features):
    """
    Computes spatio-temporal features from given features.

    Parameters:
    features (torch.Tensor): Input features to process.

    Returns:
    torch.Tensor: Spatio-temporal features.
    """

    # Extract the dimensions of the features
    t, s, c = features.shape

    # Compute temporal tokens as the mean along the time axis
    temporal_tokens = torch.mean(features, dim=1)

    # Padding size calculation
    padding_size = 100 - t

    # Pad temporal tokens if necessary
    if padding_size > 0:
        padding = torch.zeros(padding_size, c, device=features.device)
        temporal_tokens = torch.cat((temporal_tokens, padding), dim=0)

    # Compute spatial tokens as the mean along the spatial axis
    spatial_tokens = torch.mean(features, dim=0)

    # Concatenate temporal and spatial tokens and cast to half precision
    concat_tokens = torch.cat([temporal_tokens, spatial_tokens], dim=0).half()

    return concat_tokens


def video_chatgpt_infer(
    conv_templates,
    SeparatorStyle,
    KeywordsStoppingCriteria,
    video_frames,
    question,
    conv_mode,
    model,
    vision_tower,
    tokenizer,
    image_processor,
    video_token_len,
    device="cuda",
    torch_dtype=torch.float16,
    do_sample=True,
    temperature=0.2,
    max_new_tokens=1024,
    top_p=0.9,
    **kwargs,
):
    """
    Run inference using the Video-ChatGPT model.

    Parameters:
    sample : Initial sample
    video_frames (torch.Tensor): Video frames to process.
    question (str): The question string.
    conv_mode: Conversation mode.
    model: The pretrained Video-ChatGPT model.
    vision_tower: Vision model to extract video features.
    tokenizer: Tokenizer for the model.
    image_processor: Image processor to preprocess video frames.
    video_token_len (int): The length of video tokens.

    Returns:
    dict: Dictionary containing the model's output.
    """

    # Prepare question string for the model
    if model.get_model().vision_config.use_vid_start_end:
        qs = (
            question
            + "\n"
            + DEFAULT_VID_START_TOKEN
            + DEFAULT_VIDEO_PATCH_TOKEN * video_token_len
            + DEFAULT_VID_END_TOKEN
        )
    else:
        qs = question + "\n" + DEFAULT_VIDEO_PATCH_TOKEN * video_token_len

    # Prepare conversation prompt
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    # Tokenize the prompt
    inputs = tokenizer([prompt])

    # Preprocess video frames and get image tensor
    image_tensor = image_processor.preprocess(video_frames, return_tensors="pt")[
        "pixel_values"
    ]

    # Move image tensor to GPU and reduce precision to half
    image_tensor = image_tensor.half().to(device=device, dtype=torch_dtype)

    # Generate video spatio-temporal features
    with torch.no_grad():
        image_forward_outs = vision_tower(image_tensor, output_hidden_states=True)
        frame_features = image_forward_outs.hidden_states[-2][
            :, 1:
        ]  # Use second to last layer as in LLaVA
    video_spatio_temporal_features = get_spatio_temporal_features_torch(frame_features)

    # Move inputs to GPU
    input_ids = torch.as_tensor(inputs.input_ids).to(device=device)

    # Define stopping criteria for generation
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)

    # Run model inference
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            video_spatio_temporal_features=video_spatio_temporal_features.unsqueeze(0),
            do_sample=do_sample,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            stopping_criteria=[stopping_criteria],
            **kwargs,
        )

    # Check if output is the same as input
    n_diff_input_output = (
        (input_ids != output_ids[:, : input_ids.shape[1]]).sum().item()
    )
    if n_diff_input_output > 0:
        print(
            f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
        )

    # Decode output tokens
    outputs = tokenizer.batch_decode(
        output_ids[:, input_ids.shape[1] :], skip_special_tokens=True
    )[0]

    # Clean output string
    outputs = outputs.strip().rstrip(stop_str).strip()

    return outputs


class VideoChatGptHF(
    BaseHFModel[
        VideoChatGptHFInitConfig,
        VideoChatGptHFSamplingConfig,
        VideoChatGptHFConfig,
        ProcessorMixin,
    ]
):
    processor: AutoProcessor | None
    model: PreTrainedModel | None
    image_processor: CLIPImageProcessor | None

    def __init__(
        self, configs: VideoChatGptHFConfig | str = VideoChatGptHFConfig(), **kwargs
    ):
        self.processor = None
        self.image_processor = None

        super().__init__(configs, **kwargs)

        local_repo_path = self.init_config.local_repo_path
        assert local_repo_path is not None, "local_repo_path must be provided"
        sys.path.insert(0, local_repo_path)

        try:
            from video_chatgpt.eval.model_utils import initialize_model
            from video_chatgpt.video_conversation import conv_templates, SeparatorStyle
            from video_chatgpt.model.utils import KeywordsStoppingCriteria

        except Exception as err:
            logging.critical(
                "Please first install requirements and set the root path to use Video-ChatGPT. \
                Follow the instructions at https://github.com/mbzuai-oryx/Video-ChatGPT."
            )
            raise err

        base_model_path = snapshot_download("mmaaz60/LLaVA-7B-Lightening-v1-1")
        projection_path = snapshot_download("MBZUAI/Video-ChatGPT-7B")
        projection_name = "video_chatgpt-7B.bin"
        projection_path = os.path.join(projection_path, projection_name)
        model, vision_tower, tokenizer, image_processor, video_token_len = (
            initialize_model(base_model_path, projection_path)
        )
        if isinstance(image_processor, tuple):
            image_processor = image_processor[0]

        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.video_token_len = video_token_len
        self.kwargs = kwargs
        self.vision_tower = vision_tower

        self.conv_templates = conv_templates
        self.SeparatorStyle = SeparatorStyle
        self.KeywordsStoppingCriteria = KeywordsStoppingCriteria

    @staticmethod
    def get_info() -> ModelInfo:
        return ModelInfo(
            id="MBZUAI/Video-ChatGPT-7B",
            modality="video",
            short_model_size="7B",
            full_model_size=None,
            short_name="Video-ChatGPT",
            long_name="Video-ChatGPT",
            link="https://huggingface.co/MBZUAI/Video-ChatGPT-7B",
            description="Video-ChatGPT is a video-language model",
        )

    @staticmethod
    def _load_tokenizer_static(init_config):
        return None

    @staticmethod
    def _load_processor_static(init_config) -> Optional[AutoProcessor]:
        return None

    @staticmethod
    def _initialize_model_static(
        init_config: VideoChatGptHFInitConfig,
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
            # Only supports single video input
            video_frames = load_video(
                video_list[0],
                n_clips=1,
                max_frames=max_frames,
                video_framerate=target_fps,
                s=None,
                e=None,
            )

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

        conv_mode = "video-chatgpt_v1"
        outputs_list = []
        for prompt, video_frames in zip(prompt_list, video_frames_list):
            # input_ids = input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids
            # video_frames = video_frames.unsqueeze(0) if video_frames.dim() == 3 else video_frames

            # Run inference on the video and questions
            output = video_chatgpt_infer(
                self.conv_templates,
                self.SeparatorStyle,
                self.KeywordsStoppingCriteria,
                video_frames,
                prompt,
                conv_mode,
                self.model,
                self.vision_tower,
                self.tokenizer,
                self.image_processor,
                self.video_token_len,
                device=self.init_config.device,
                torch_dtype=self.init_config.torch_dtype,
                do_sample=self.sampling_config.do_sample,
                temperature=self.sampling_config.temperature,
                max_new_tokens=self.sampling_config.max_new_tokens,
                top_p=self.sampling_config.top_p,
            )

            outputs_list.append(output)
            print("generate completed ... ")

        return outputs_list


if __name__ == "__main__":
    model = VideoChatGptHF()
    test_model(model)
