from typing import Type

from driving_vllm_arena.inference.configs.app import TARGET_FPS
from driving_vllm_arena.inference.models.base import BaseInferenceModel
from inspect import signature
from typing import get_args, get_origin


def _model_expects_raw_paths(model) -> bool:
    """Check if model's generate method expects raw video paths (str) or VideoAsset objects."""
    sig = signature(model.generate)
    videos_type = sig.parameters["videos"].annotation

    # Unwrap nested types (e.g., List[List[Union[str, VideoAsset]]])
    for _ in range(2):  # Unwrap outer List and inner List
        videos_type = get_args(videos_type)[0]

    return str in get_args(videos_type)


def test_model(model: BaseInferenceModel):
    """
    Test a VLLM model with example videos.
    Args:
        model: The VLLM model to test
    """
    # Example video paths
    video_paths = [
        # "/home/username/Projects/DrivingVllmBench/videos/bird.mp4",
        "/home/username/Projects/DrivingVllmBench/videos/5s/96898040.mp4",
        "/home/username/Projects/DrivingVllmBench/videos/5s/96898041.mp4",
        "/home/username/Projects/DrivingVllmBench/videos/5s/96898042.mp4",
        # "/home/username/Projects/DrivingVllmBench/videos/5s/96898043.mp4",
        # "/home/username/Projects/DrivingVllmBench/videos/5s/96898044.mp4",
        # "/home/username/Projects/DrivingVllmBench/videos/5s/96898045.mp4",
        # "/home/username/Projects/DrivingVllmBench/videos/5s/96898046.mp4",
        # "/home/username/Projects/DrivingVllmBench/videos/5s/96898047.mp4",
        # "/home/username/Projects/DrivingVllmBench/videos/5s/96898048.mp4",
        # "/home/username/Projects/DrivingVllmBench/videos/5s/96898049.mp4",
    ]

    # Load videos based on use_raw_paths flag
    if _model_expects_raw_paths(model):
        videos = [[raw_path] for raw_path in video_paths]
    else:
        from vllm.assets.video import VideoAsset

        videos = [[VideoAsset(name=path, num_frames=16)] for path in video_paths]  # type: ignore

    # Example prompts - one per video
    prompts = ["Please describe this video in details."] * len(videos)

    # Generate outputs
    outputs = model.generate(prompts, videos, target_fps=TARGET_FPS)

    # Print results
    for i, output in enumerate(outputs):
        print(f"\nVideo {i+1} ({video_paths[i]}):")
        if hasattr(output, "outputs") and output.outputs:
            print(f"Generated text: {output.outputs[0].text}\n")
        elif hasattr(output, "text"):
            print(f"Generated text: {output.text}\n")
        else:
            print(f"Generated output: {str(output)}\n")
        print("-" * 50)


def test_model_with_questions(
    model: BaseInferenceModel,
    question_bank_path: str = "outputs/benchmarking/question_bank/driving_video_benchmark_samples.json",
    group_index: int = 0,
    max_retries: int = 1,
):
    # Import the function here instead to avoid circular imports
    from driving_vllm_arena.benchmarking.evaluate import process_specific_question_group

    process_specific_question_group(
        model, question_bank_path, group_index=group_index, max_retries=max_retries
    )
