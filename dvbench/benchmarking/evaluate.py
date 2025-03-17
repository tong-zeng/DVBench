import json
import os
from enum import Enum, auto
from pathlib import Path
import logging
from tqdm import tqdm
from typing import Optional, List, TypeVar, Any

from dvbench.inference.configs.app import TARGET_FPS
from dvbench.inference.models.base import BaseInferenceModel
from dvbench.question_bank_gen.gen_question_bank import load_from_jsonl
from dvbench.inference.configs.app import PROJECT_ROOT
from inspect import signature
from typing import get_args, get_origin
from dvbench.inference.configs.app import TARGET_FRAMES_NUM

logger = logging.getLogger(__name__)
M = TypeVar("M", bound=BaseInferenceModel)


class DomainKnowledgePlacement(Enum):
    BEFORE_QUESTION = auto()
    AFTER_QUESTION = auto()
    AFTER_OPTIONS = auto()


class AccuracyMode(Enum):
    INDIVIDUAL = auto()
    GROUP = auto()


def load_question_bank_sync(file_path: str) -> List[dict]:
    """Load questions from JSON or JSONL file."""
    return (
        load_from_jsonl(file_path)
        if file_path.endswith(".jsonl")
        else json.load(open(file_path, "r"))
    )


def construct_prompt_text(
    question: dict, domain_knowledge_placement: Optional[DomainKnowledgePlacement]
) -> str:
    """Construct prompt text with optional domain knowledge placement."""
    question_text = question["question"]
    options = "\n".join([f"{opt}" for opt in question["options"]])
    domain_knowledge = question.get("domain_knowledge", "")

    parts = {
        "domain": f"Domain Knowledge: {domain_knowledge}\n\n"
        if domain_knowledge
        else "",
        "question": f"Question: {question_text}\n\n",
        "options": f"Options:\n{options}",
    }

    if domain_knowledge_placement == DomainKnowledgePlacement.BEFORE_QUESTION:
        return parts["domain"] + parts["question"] + parts["options"]
    elif domain_knowledge_placement == DomainKnowledgePlacement.AFTER_QUESTION:
        return parts["question"] + parts["domain"] + parts["options"]
    elif domain_knowledge_placement == DomainKnowledgePlacement.AFTER_OPTIONS:
        return parts["question"] + parts["options"] + "\n\n" + parts["domain"]
    return parts["question"] + parts["options"]


def _model_expects_raw_paths(model) -> bool:
    """Check if model's generate method expects raw video paths (str) or VideoAsset objects."""
    sig = signature(model.generate)
    videos_type = sig.parameters["videos"].annotation

    # Unwrap nested types (e.g., List[List[Union[str, VideoAsset]]])
    for _ in range(2):  # Unwrap outer List and inner List
        videos_type = get_args(videos_type)[0]

    return str in get_args(videos_type)


def process_batch(
    model: M,
    batch: List[dict],
    domain_knowledge_placement: Optional[DomainKnowledgePlacement],
    max_retries: int,
) -> List[dict]:
    """Process a batch of questions and return results."""
    prompts = []
    videos = []
    for question in batch:
        prompts.append(construct_prompt_text(question, domain_knowledge_placement))

        if _model_expects_raw_paths(model):
            videos.append(
                [f"{PROJECT_ROOT}/videos/Reduced_Videos/{question['video_id']}.mp4"]
            )
        else:
            from vllm.assets.video import VideoAsset

            videos.append(
                [
                    VideoAsset(
                        name=f"{PROJECT_ROOT}/videos/Reduced_Videos/{question['video_id']}.mp4",  # type: ignore
                        num_frames=TARGET_FRAMES_NUM,
                    )
                ]
            )

    for attempt in range(max_retries):
        try:
            outputs = model.generate(prompts, videos, target_fps=TARGET_FPS)

            for question, prompt, output in zip(batch, prompts, outputs):
                question["prompt"] = prompt
                question["response"] = (
                    output.outputs[0].text
                    if hasattr(output, "outputs") and output.outputs
                    else output.text
                    if hasattr(output, "text")
                    else str(output)
                )
            return batch
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            if attempt == max_retries - 1:
                for question, prompt in zip(batch, prompts):
                    question["prompt"] = prompt
                    question["response"] = (
                        f"Error: Request failed after {max_retries} attempts"
                    )
    return batch


def get_output_path(
    model_name: str,
    question_bank_path: str,
    ai_response_output_dir: Path | str,
    placement: Optional[DomainKnowledgePlacement],
    start_pos: int,
    end_pos: int,
) -> str:
    """
    Generate output path for results within a subfolder named after the question bank.

    Args:
        model: The model being evaluated
        question_bank_path: Path to the question bank file
        placement: Domain knowledge placement strategy
        start_pos: Starting position in the question bank
        end_pos: Ending position in the question bank
        ai_response_output_dir: Base directory for saving AI model responses. If None, uses default location

    Example:
    If question_bank_path is '/path/to/question_bank/bank_100.jsonl',
    output will be '{ai_response_output_dir}/bank_100/ai_responses_bank_100_0_102_model_ModelName_domain_knowledge_placement_none.jsonl'
    """
    question_bank_path_obj = Path(question_bank_path)
    bank_name = question_bank_path_obj.stem
    placement_str = placement.name.lower() if placement else "none"
    end_str = str(end_pos) if end_pos != -1 else "all"

    # Use provided output directory or default to question_bank's parent's ai_responses folder
    if ai_response_output_dir is None:
        output_dir = question_bank_path_obj.parent.parent / "ai_responses" / bank_name
    else:
        output_dir = Path(ai_response_output_dir) / bank_name / model_name

    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"ai_responses_{bank_name}_{start_pos}_{end_str}_model_{model_name}_domain_knowledge_{placement_str}.jsonl"
    return str(output_dir / filename)


def process_questions(
    model: M,
    questions: List[dict],
    output_path: str,
    domain_knowledge_placement: Optional[DomainKnowledgePlacement],
    batch_size: int = 4,
    max_retries: int = 3,
    start_pos: int = 0,
) -> None:
    """Process questions in batches and save results."""
    mode = "r+" if (start_pos > 0 and os.path.exists(output_path)) else "w"

    with open(output_path, mode) as f:
        if start_pos > 0:
            for _ in range(start_pos):
                f.readline()
            f.truncate(f.tell())

        for i in tqdm(
            range(0, len(questions), batch_size), desc="Processing questions"
        ):
            batch = questions[i : i + batch_size]
            processed_batch = process_batch(
                model, batch, domain_knowledge_placement, max_retries
            )

            for question in processed_batch:
                f.write(json.dumps(question, ensure_ascii=False) + "\n")
                f.flush()


def evaluate_model(
    model: M,
    model_name: str,
    question_bank_path: str,
    ai_response_output_dir: Path | str,
    domain_knowledge_placement: List[DomainKnowledgePlacement | None],
    batch_size: int = 4,
    start_pos: int = 0,
    end_pos: int = -1,
):
    """Generate responses for multiple domain knowledge placements."""
    # Load and flatten questions from the question bank
    questions = []
    for group in load_question_bank_sync(question_bank_path):
        questions.extend(group["question_group"])

    # Default to just None placement if not specified
    placements = domain_knowledge_placement or list(DomainKnowledgePlacement) + [None]

    for placement in placements:
        logger.info(f"Generating responses for placement: {placement}")
        output_path = get_output_path(
            model_name,
            question_bank_path,
            ai_response_output_dir,
            placement,
            start_pos,
            end_pos,
        )
        process_questions(
            model,
            questions[start_pos : end_pos or None],
            output_path,
            placement,
            batch_size=batch_size,
            start_pos=start_pos,
        )

    logger.info("All experiments completed.")


if __name__ == "__main__":
    from dvbench.inference.models.index import SUPPORTED_MODELS

    model = SUPPORTED_MODELS["mini_cpm_v"]()
    project_root = Path(__file__).parent.parent.parent
    question_bank_path = (
        project_root
        / "all_experiments/table1_main_performance/question_bank/question_bank_1000.jsonl"
    )

    if not question_bank_path.exists():
        raise FileNotFoundError(f"Question bank not found: {question_bank_path}")

    # Example of using custom output directory
    ai_response_output_dir = (
        project_root / "all_experiments/table1_main_performance/ai_responses"
    )
    model_name = "chat_uni_vi_7b"

    evaluate_model(
        model,
        model_name,
        str(question_bank_path),
        ai_response_output_dir,
        domain_knowledge_placement=[None],
        batch_size=1,
        start_pos=0,
        end_pos=-1,
    )
