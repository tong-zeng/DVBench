# Initialize empty dictionaries

_VLLM_MODELS = {}
_HF_MODELS = {}

# Try importing vLLM models, but don't fail if vllm is not installed
try:
    from dvbench.inference.models.vllm_models.mini_cpm_v import MiniCPMVvLLM

    _VLLM_MODELS.update({"mini_cpm_v": MiniCPMVvLLM})
except ImportError:
    # MiniCPMVvLLM not installed - skip MiniCPMVvLLM
    pass

try:
    from dvbench.inference.models.vllm_models.qwen2_vl import Qwen2VLvLLM
    from dvbench.inference.models.vllm_models.llava_next_video import (
        LLaVANextVideovLLM,
    )
    from dvbench.inference.models.vllm_models.llava_onevision import (
        LLaVAOneVisionvLLM,
    )

    # Only populate vLLM models if imports succeed
    _VLLM_MODELS.update(
        {
            "mini_cpm_v": MiniCPMVvLLM,
            "llava_next_video_7b": LLaVANextVideovLLM,
            "llava_next_video_34b": LLaVANextVideovLLM,
            "llava_onevision_0_5b": LLaVAOneVisionvLLM,
            "llava_onevision_7b": LLaVAOneVisionvLLM,
            "llava_onevision_72b": LLaVAOneVisionvLLM,
            "qwen2_vl_2b": Qwen2VLvLLM,
            "qwen2_vl_7b": Qwen2VLvLLM,
            "qwen2_vl_72b": Qwen2VLvLLM,
        }
    )

except ImportError:
    # vllm not installed - skip vLLM models
    pass


try:
    from dvbench.inference.models.hf_models.chat_uni_vi import ChatUniViHF

    _HF_MODELS.update(
        {
            "chat_uni_vi_7b": ChatUniViHF,
            "chat_uni_vi_13b": ChatUniViHF,
        }
    )
except ImportError:
    # ChatUniViHF not installed - skip ChatUniViHF
    pass

try:
    from dvbench.inference.models.hf_models.llama_vid import LLaMAVidHF

    _HF_MODELS.update(
        {
            "llama_vid_7b": LLaMAVidHF,
            "llama_vid_13b": LLaMAVidHF,
        }
    )
except ImportError:
    # LLaVaVidHF not installed - skip LLaVaVidHF
    pass

try:
    from dvbench.inference.models.hf_models.pllava import PLLaVAHF

    _HF_MODELS.update(
        {
            "pllava_7b": PLLaVAHF,
            "pllava_13b": PLLaVAHF,
            "pllava_34b": PLLaVAHF,
        }
    )
except ImportError:
    # PLLaVAHF not installed - skip PLLaVAHF
    pass

try:
    from dvbench.inference.models.hf_models.video_chatgpt import (
        VideoChatGptHF,
    )

    _HF_MODELS.update(
        {
            "video_chatgpt_7b": VideoChatGptHF,
        }
    )
except ImportError:
    # VideoChatGptHF not installed - skip VideoChatGptHF
    pass

try:
    from dvbench.inference.models.hf_models.video_llava import VideoLLaVAHF

    _HF_MODELS.update(
        {
            "video_llava_7b": VideoLLaVAHF,
        }
    )
except ImportError:
    # VideoLLaVAHF not installed - skip VideoLLaVAHF
    pass

# Combine all available models
SUPPORTED_MODELS = {}
model_groups = [_HF_MODELS, _VLLM_MODELS]

for model_group_dict in model_groups:
    SUPPORTED_MODELS.update(model_group_dict)

if __name__ == "__main__":
    print(f"Supported VLLM models: {SUPPORTED_MODELS}")


# driving_vllm_arena/inference/models/hf_models/hf_model_configs/llama_vid/processor/clip-patch14-224

# driving_vllm_arena/inference/models/hf_models/hf_model_configs/llava_vid/processor/clip-patch14-224
