from typing import List, Optional, Any, Union
from transformers import ProcessorMixin, PreTrainedModel
from pydantic import ConfigDict
from transformers import Qwen2VLForConditionalGeneration, Qwen2VLProcessor, BatchFeature
from qwen_vl_utils import process_vision_info
import torch
from dvbench.inference.models.base import ModelInfo, ModelConfig
from dvbench.inference.models.hf_models.base_hf_model import BaseHFModel

from dvbench.inference.models.hf_models.hf_model_configs.config_qwen2_vl import (
    Qwen2VLHFInitConfig,
    Qwen2VLHFSamplingConfig,
    Qwen2VLHFModelConfig
)
from dvbench.inference.models.model_test import test_model
from qwen_vl_utils import process_vision_info
from dvbench.inference.configs.app import TARGET_FPS

class Qwen2VLHF(BaseHFModel[Qwen2VLHFInitConfig, Qwen2VLHFSamplingConfig, Qwen2VLHFModelConfig, Qwen2VLProcessor]):
    model_config = ConfigDict(
        extra='forbid',
        arbitrary_types_allowed=True
    )

    model: Qwen2VLForConditionalGeneration

    def __init__(
        self,
        configs: Qwen2VLHFModelConfig|str= Qwen2VLHFModelConfig(),
        **kwargs
    ):
        super().__init__(configs, **kwargs)
        
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
            description="Qwen2 VL is a multimodal model that can process both images and videos"
        )
    
    @staticmethod
    def _initialize_model_static(init_config: Qwen2VLHFInitConfig) -> PreTrainedModel:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            init_config.pretrained_model_name_or_path,
            trust_remote_code=init_config.trust_remote_code,
            torch_dtype=init_config.torch_dtype,
        )
        model.to(init_config.device) # type: ignore
        return model

    def prepare_inputs(
            self, 
            prompts: Union[str, List[str]], 
            videos: Optional[List[List[str]]] = None,
            target_fps: int = TARGET_FPS
        ) -> BatchFeature:
        
        # Normalize prompts to List[str]
        if isinstance(prompts, str):
            prompts = [prompts]
            
        # Validate videos type and structure
        if videos is not None:
            if not (isinstance(videos, list) and 
                   all(isinstance(sublist, list) and 
                       all(isinstance(v, str) for v in sublist) 
                       for sublist in videos)):
                raise ValueError("videos must be List[List[str]]")
            
            # Check length match
            if len(prompts) != len(videos):
                raise ValueError(f"Number of prompts ({len(prompts)}) must match number of video lists ({len(videos)})")

        # Create empty video lists if videos is None
        conversations = []
        texts = []
        videos = videos or [[] for _ in prompts]
        for (prompt, video_list) in zip(prompts, videos):
            # Use pil_images property to get the frames as PIL Images
            
            # The process_vision_info only supports single video input
            conversation = [{
                "role": "user",
                "content": [
                    {"type": "video", "video": video_list[0], "fps": target_fps},
                    {"type": "text", "text": prompt},
                ],
            }]

            prompt_text = self.processor.apply_chat_template( # type: ignore
                conversation,
                add_generation_prompt=True,
                tokenize=False
            )

            conversations.append(conversation)
            texts.append(prompt_text)
        image_inputs, video_inputs = process_vision_info(conversations)
        inputs = self.processor(
            text=texts,
            images=image_inputs, # type: ignore
            videos=video_inputs, # type: ignore
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.init_config.device)
        return inputs

    def generate(self, prompts: Union[str, List[str]], videos: Optional[List[List[str]]] = None, **kwargs) -> List[Any]:
        """Generate outputs for a single prompt and video."""
        inputs = self.prepare_inputs(prompts, videos)
        with torch.inference_mode():            
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.sampling_config.max_tokens,
                top_p=self.sampling_config.top_p,
                top_k=self.sampling_config.top_k,
                temperature=self.sampling_config.temperature,
                )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode( # type: ignore
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return output_text


if __name__ == "__main__":
    model = Qwen2VLHF()
    test_model(model)