import os
from typing import Optional
from enum import Enum, auto
import json
import logging

logger = logging.getLogger(__name__)


class DomainKnowledgePlacement(Enum):
    BEFORE_QUESTION = auto()
    AFTER_QUESTION = auto()
    AFTER_OPTIONS = auto()


class AccuracyMode(Enum):
    INDIVIDUAL = auto()
    GROUP = auto()


def get_output_path(
    domain_knowledge_placement: Optional[DomainKnowledgePlacement],
    start_at_pos: int,
    end_at_pos: int,
) -> str:
    placement_str = (
        domain_knowledge_placement.name.lower()
        if domain_knowledge_placement
        else "no_domain_knowledge"
    )
    return f"outputs/benchmarking/ai_responses/ai_responses_for_questions_{end_at_pos - start_at_pos}_domain_knowledge_placement_{placement_str}.jsonl"


def calculate_and_print_accuracy(jsonl_path, mode=AccuracyMode.GROUP):
    questions = []
    with open(jsonl_path, "r") as f:
        for line in f:
            questions.append(json.loads(line.strip()))

    accuracy, correct, total = calculate_accuracy(questions, mode=mode)
    logger.info(
        f"Mode: {mode.name}, Accuracy: {accuracy:.2%}, Correct: {correct}, Total: {total}"
    )
    return accuracy


def calculate_accuracy(question_group_list, mode=AccuracyMode.INDIVIDUAL):
    correct = 0
    total = 0

    for question_group in question_group_list:
        if mode == AccuracyMode.INDIVIDUAL:
            for question in question_group["question_group"]:
                response = question["response"].strip()
                ground_truth = question["ground_truth"].strip()
                logger.debug(f"Response: {response}")
                logger.debug(f"Ground Truth: {ground_truth}")
                if response == ground_truth or ground_truth.startswith(response):
                    correct += 1
                    logger.debug("Correct!")
                else:
                    logger.debug("Incorrect.")
                logger.debug("---")  # Separator for readability
                total += 1
        elif mode == AccuracyMode.GROUP:
            group_correct = all(
                question["response"].strip() == question["ground_truth"].strip()
                or question["ground_truth"]
                .strip()
                .startswith(question["response"].strip())
                for question in question_group["question_group"]
            )
            if group_correct:
                correct += 1
                logger.debug("Group Correct!")
            else:
                logger.debug("Group Incorrect.")
            logger.debug("---")  # Separator for readability
            total += 1
        else:
            raise ValueError(
                "Invalid mode. Use AccuracyMode.INDIVIDUAL or AccuracyMode.GROUP."
            )

    return (correct / total if total > 0 else 0, correct, total)


def calculate_performance(
    domain_knowledge_placement: Optional[DomainKnowledgePlacement],
    start_at_pos: int,
    end_at_pos: int,
):
    output_path = get_output_path(domain_knowledge_placement, start_at_pos, end_at_pos)

    # Add a check to see if the file exists
    if not os.path.exists(output_path):
        print(f"Warning: File not found: {output_path}")
        return {
            "domain_knowledge_placement": domain_knowledge_placement.name.lower()
            if domain_knowledge_placement
            else "no_domain_knowledge",
            "individual_accuracy": None,
            "group_accuracy": None,
            "output_path": output_path,
            "error": "File not found",
        }

    individual_accuracy = calculate_and_print_accuracy(
        output_path, mode=AccuracyMode.INDIVIDUAL
    )
    group_accuracy = calculate_and_print_accuracy(output_path, mode=AccuracyMode.GROUP)

    placement_str = (
        domain_knowledge_placement.name.lower()
        if domain_knowledge_placement
        else "no_domain_knowledge"
    )

    return {
        "domain_knowledge_placement": placement_str,
        "individual_accuracy": individual_accuracy,
        "group_accuracy": group_accuracy,
        "output_path": output_path,
    }


def calculate_all_performances(start_at_pos: int = 0, end_at_pos: int = 102):
    placements = list(DomainKnowledgePlacement) + [None]
    results = []
    for placement in placements:
        print(f"Calculating performance for {placement}")
        result = calculate_performance(placement, start_at_pos, end_at_pos)
        results.append(result)
        print(f"\nResults for {result['domain_knowledge_placement']}:")
        if "error" in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Individual Accuracy: {result['individual_accuracy']:.2%}")
            print(f"Group Accuracy: {result['group_accuracy']:.2%}")
        print(f"Output path: {result['output_path']}")
        print("-" * 50)
    return results


if __name__ == "__main__":
    start_at_pos = 0
    end_at_pos = 102
    # Calculate performance for all configurations
    results = calculate_all_performances(start_at_pos, end_at_pos)

    # outputs/benchmarking/ai_responses/ai_responses_for_questions_102_domain_knowledge_placement_before_question.jsonl
    # calculate_all_performances()
