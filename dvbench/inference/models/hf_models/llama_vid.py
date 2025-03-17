from typing import List, Optional, Any, Union
from pydantic import ConfigDict
from transformers import (
    PreTrainedModel,
    AutoModelForCausalLM,
    AutoProcessor,
    ProcessorMixin,
)
import torch
import os
import numpy as np
from PIL import Image
from decord import VideoReader, cpu

from dvbench.inference.models.base import ModelInfo
from dvbench.inference.models.hf_models.base_hf_model import BaseHFModel
from dvbench.inference.models.hf_models.hf_model_configs.config_llama_vid import (
    LLaMAVidHFInitConfig,
    LLaMAVidHFSamplingConfig,
    LLaMAVidHFConfig,
)
from dvbench.inference.models.model_test import test_model
from dvbench.inference.configs.app import MAX_FRAMES, TARGET_FPS

from transformers.generation.stopping_criteria import StoppingCriteriaList

from llamavid.conversation import conv_templates, SeparatorStyle
from llamavid.model import LlavaLlamaAttForCausalLM

from llava.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria
from llamavid.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llamavid.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from llamavid.constants import (
    DEFAULT_IMAGE_PATCH_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from huggingface_hub import snapshot_download

import pickle
import json
import csv
import pandas as pd
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),  # type: ignore
        ):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):  # type: ignore
            return float(obj)
        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):  # type: ignore
            return {"real": obj.real, "imag": obj.imag}
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (np.void)):
            return None
        return json.JSONEncoder.default(self, obj)


# LOAD & DUMP
def dump(data, f, **kwargs):
    def dump_pkl(data, pth, **kwargs):
        pickle.dump(data, open(pth, "wb"))

    def dump_json(data, pth, **kwargs):
        json.dump(data, open(pth, "w"), indent=4, ensure_ascii=False, cls=NumpyEncoder)

    def dump_jsonl(data, f, **kwargs):
        lines = [json.dumps(x, ensure_ascii=False, cls=NumpyEncoder) for x in data]
        with open(f, "w", encoding="utf8") as fout:
            fout.write("\n".join(lines))

    def dump_xlsx(data, f, **kwargs):
        data.to_excel(f, index=False, engine="xlsxwriter")

    def dump_csv(data, f, quoting=csv.QUOTE_ALL):
        data.to_csv(f, index=False, encoding="utf-8", quoting=quoting)

    def dump_tsv(data, f, quoting=csv.QUOTE_ALL):
        data.to_csv(f, sep="\t", index=False, encoding="utf-8", quoting=quoting)

    handlers = dict(
        pkl=dump_pkl,
        json=dump_json,
        jsonl=dump_jsonl,
        xlsx=dump_xlsx,
        csv=dump_csv,
        tsv=dump_tsv,
    )
    suffix = f.split(".")[-1]
    return handlers[suffix](data, f, **kwargs)


def load(f, fmt=None):
    def load_pkl(pth):
        return pickle.load(open(pth, "rb"))

    def load_json(pth):
        return json.load(open(pth, "r", encoding="utf-8"))

    def load_jsonl(f):
        lines = open(f, encoding="utf-8").readlines()
        lines = [x.strip() for x in lines]
        if lines[-1] == "":
            lines = lines[:-1]
        data = [json.loads(x) for x in lines]
        return data

    def load_xlsx(f):
        return pd.read_excel(f)

    def load_csv(f):
        return pd.read_csv(f)

    def load_tsv(f):
        return pd.read_csv(f, sep="\t")

    handlers = dict(
        pkl=load_pkl,
        json=load_json,
        jsonl=load_jsonl,
        xlsx=load_xlsx,
        csv=load_csv,
        tsv=load_tsv,
    )
    if fmt is not None:
        return handlers[fmt](f)

    suffix = f.split(".")[-1]
    return handlers[suffix](f)


def download_file(url, filename=None):
    import urllib.request
    from tqdm import tqdm

    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    if filename is None:
        filename = url.split("/")[-1]

    try:
        with DownloadProgressBar(
            unit="B", unit_scale=True, miniters=1, desc=url.split("/")[-1]
        ) as t:
            urllib.request.urlretrieve(url, filename=filename, reporthook=t.update_to)
    except Exception as e:
        import logging

        logging.warning(f"{type(e)}: {e}")
        # Handle Failed Downloads from huggingface.co
        if "huggingface.co" in url:
            url_new = url.replace("huggingface.co", "hf-mirror.com")
            try:
                download_file(url_new, filename)
                return filename
            except Exception as e:
                logging.warning(f"{type(e)}: {e}")
                raise Exception(f"Failed to download {url}")
        else:
            raise Exception(f"Failed to download {url}")

    return filename


def load_video(video_path, setting_fps):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    fps = round(vr.get_avg_fps())
    frame_idx = [i for i in range(0, total_frame_num, int(fps / setting_fps))]
    spare_frames = vr.get_batch(frame_idx).asnumpy()
    return spare_frames


def change_file(file_path, mm_vision_tower):
    org_data = load(file_path)
    org_data["image_processor"] = (  # type: ignore
        "/home/username/Projects/DrivingVllmBench/driving_vllm_arena/inference/models/hf_models/hf_model_configs/llama_vid/processor/clip-patch14-224"
    )
    org_data["mm_vision_tower"] = mm_vision_tower  # type: ignore
    dump(org_data, file_path)


class LLaMAVidHF(
    BaseHFModel[
        LLaMAVidHFInitConfig, LLaMAVidHFSamplingConfig, LLaMAVidHFConfig, ProcessorMixin
    ]
):
    processor: AutoProcessor | None
    model: PreTrainedModel
    image_processor: Any | None

    def __init__(self, configs: LLaMAVidHFConfig | str = LLaMAVidHFConfig(), **kwargs):
        self.processor = None
        self.image_processor = None

        super().__init__(configs, image_processor=None, **kwargs)
        self._setup_special_tokens()
        self._setup_vision_tower()
        self._setup_attention_modules()
        self._setup_context_len()

    @staticmethod
    def _setup_model_config(pretrained_model_name_or_path):
        eva_vit_g_url = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth"
        true_model_path = snapshot_download(pretrained_model_name_or_path)
        eva_vit_path = os.path.join(true_model_path, "eva_vit_g.pth")
        if not os.path.exists(eva_vit_path):
            download_file(eva_vit_g_url, eva_vit_path)
        config_path = os.path.join(true_model_path, "config.json")
        change_file(config_path, eva_vit_path)

    def _setup_special_tokens(self):
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
        vision_tower = self.model.get_vision_tower()  # type: ignore
        if not vision_tower.is_loaded:
            vision_tower.load_model()
        vision_tower = vision_tower.to(
            device=self.init_config.device, dtype=self.init_config.torch_dtype
        )
        self.image_processor = vision_tower.image_processor

    def _setup_attention_modules(self):
        # initialize attention modules
        print("_setup_attention_modules setup attention modules started ... ")
        true_model_path = snapshot_download(
            self.init_config.pretrained_model_name_or_path
        )
        print("_setup_attention_modules true_model_path completed ... ")
        self.model.config.model_path = true_model_path
        self.model.get_model().initialize_attention_modules(
            self.model.config, for_eval=True
        )  # type: ignore

    def _setup_context_len(self):
        self.context_len = getattr(
            self.model.config, "max_sequence_length", 4096
        )  # Changed from 2048 to 4096

    @staticmethod
    def get_info() -> ModelInfo:
        return ModelInfo(
            id="YanweiLi/llama-vid-7b-full-224-video-fps-1",
            modality="video",
            short_model_size="7B",
            full_model_size=None,
            short_name="LLaVA-Vid",
            long_name="LLaVA-Vid 7B",
            link="https://huggingface.co/YanweiLi/llama-vid-7b-full-224-video-fps-1",
            description="LLaVA-Vid is a video-language model based on LLaVA",
        )

    @staticmethod
    def _load_processor_static(init_config) -> Optional[AutoProcessor]:
        return None

    @staticmethod
    def _initialize_model_static(init_config: LLaMAVidHFInitConfig) -> PreTrainedModel:
        LLaMAVidHF._setup_model_config(init_config.pretrained_model_name_or_path)
        print("_initialize_model_static model download started ... ")
        model = LlavaLlamaAttForCausalLM.from_pretrained(
            init_config.pretrained_model_name_or_path,
            trust_remote_code=init_config.trust_remote_code,
            torch_dtype=init_config.torch_dtype,
        )
        print("_initialize_model_static model download completed ... ")
        model = model[0] if isinstance(model, tuple) else model

        # Convert string types to proper PyTorch types
        device = torch.device(init_config.device)
        model = model.to(device=device, dtype=init_config.torch_dtype)  # type: ignore
        print("_initialize_model_static model to device completed ... ")
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
            if self.model.config.mm_use_im_start_end:
                prompt = (
                    DEFAULT_IM_START_TOKEN
                    + DEFAULT_IMAGE_TOKEN
                    + DEFAULT_IM_END_TOKEN
                    + "\n"
                    + prompt
                )
            else:
                prompt = DEFAULT_IMAGE_TOKEN + "\n" + prompt

            conv = conv_templates[self.init_config.conv_mode].copy()
            conv.append_message(conv.roles[0], prompt)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt().strip("</s>")

            if os.path.exists(video_list[0]):
                video = load_video(video_list[0], target_fps)
                video = (
                    self.image_processor.preprocess(  # type: ignore
                        video,
                        return_tensors="pt",
                    )["pixel_values"]
                    .half()
                    .to(self.init_config.device)
                )  # type: ignore
                video = [video]

            input_ids = (
                tokenizer_image_token(
                    prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)  # type: ignore
                .to(self.init_config.device)
            )  # type: ignore

            input_ids_list.append(input_ids)
            video_frames_list.append(video)

        return input_ids_list, video_frames_list

    def generate(
        self,
        prompts: Union[str, List[str]],
        videos: Optional[List[List[str]]] = None,
        target_fps: int = TARGET_FPS,
        **kwargs,
    ) -> List[str]:
        # print("prepare_inputs started ... ")
        input_ids_list, video_frames_list = self.prepare_inputs(
            prompts, videos, target_fps
        )
        outputs_list = []
        # print("prepare_inputs ended ... ")

        for input_ids, video_frames, prompt in zip(
            input_ids_list, video_frames_list, prompts
        ):
            # input_ids = input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids
            # video_frames = video_frames.unsqueeze(0) if video_frames.dim() == 3 else video_frames

            conv = conv_templates[self.init_config.conv_mode].copy()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            stopping_criteria = KeywordsStoppingCriteria(
                [stop_str], self.tokenizer, input_ids
            )
            with torch.inference_mode():
                self.model.update_prompt([[prompt]])
                output_ids = self.model.generate(
                    input_ids,
                    images=video_frames,
                    do_sample=self.sampling_config.do_sample,
                    temperature=self.sampling_config.temperature,
                    # top_p=self.sampling_config.top_p,
                    # num_beams=self.sampling_config.num_beams,
                    max_new_tokens=self.sampling_config.max_new_tokens,
                    use_cache=self.sampling_config.use_cache,
                    stopping_criteria=[stopping_criteria],  # type: ignore
                    # output_scores=self.sampling_config.output_scores,
                    # return_dict_in_generate=self.sampling_config.return_dict_in_generate,
                )

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (
                (input_ids != output_ids[:, :input_token_len]).sum().item()
            )
            if n_diff_input_output > 0:
                print(
                    f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
                )
            outputs = self.tokenizer.batch_decode(
                output_ids[:, input_token_len:], skip_special_tokens=True
            )[0]
            outputs = outputs.strip()
            if outputs.endswith(stop_str):
                outputs = outputs[: -len(stop_str)]
            outputs = outputs.strip()
            outputs_list.append(outputs)
            print("generate completed ... ")

        return outputs_list


if __name__ == "__main__":
    model = LLaMAVidHF()
    test_model(model)
