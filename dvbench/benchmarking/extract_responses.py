from pathlib import Path
import json
import re
import pandas as pd

answer_pattern = re.compile(r"[\s][ABCD]\.\s*")


def is_response_correct(response: str, ground_truth: str) -> bool:
    """Soft comparison between response and ground truth"""
    response = response.strip().upper()
    ground_truth = ground_truth.strip().upper()

    answer_prefix = "Answer:".upper()
    if answer_prefix in response:
        response = response.split(answer_prefix)[1].strip()

    if response == ground_truth:
        return True

    if ground_truth.startswith(response):
        return True

    if len(ground_truth) > len(response) and (ground_truth in response):
        return True

    # Process for special cases
    matches = answer_pattern.findall(response)
    if len(matches) == 1:
        matched_answer = matches[0].strip()
        if ground_truth.startswith(matched_answer):
            print(
                f"============================\nMatch: {matched_answer}, original: {response}"
            )
            return True
    return False


def extract_responses(input_path: str) -> None:
    # Create output directory and path
    input_path = Path(input_path)  # type: ignore
    output_dir = input_path.parent / "extracted_responses"  # type: ignore
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / f"{input_path.stem}_extracted_responses.jsonl"  # type: ignore

    extracted_results = []

    # Read and process input file
    with open(input_path, "r") as f:
        for line in f:
            data = json.loads(line)
            response = data["response"].strip().upper()
            ground_truth = data["ground_truth"].strip().upper()

            if is_response_correct(response, ground_truth):
                extracted_letter = ground_truth[0]
            else:
                if response[0] in ["A", "B", "C", "D"]:
                    extracted_letter = response[0]
                else:
                    extracted_letter = None

            # Add extracted response to original data
            result = {
                "question_field": data["question_field"],
                "video_id": data["video_id"],
                "ground_truth": data["ground_truth"],
                "original_response": data["response"],
                "extracted_response_letter": extracted_letter,
            }

            extracted_results.append(result)

    # Save results
    with open(output_path, "w") as f:
        for result in extracted_results:
            f.write(json.dumps(result) + "\n")

    print(f"Extracted responses saved to: {output_path}")


def extract_all_responses(experiments_dir: str | Path) -> None:
    """
    Scan the experiments directory and extract responses from all AI response files
    Args:
        experiments_dir: Path to the experiments directory
    """
    experiments_dir = Path(experiments_dir)

    # Scan for experiment files
    for model_dir in experiments_dir.iterdir():
        if not model_dir.is_dir():
            continue

        for file_path in model_dir.glob("ai_responses_*.jsonl"):
            print(f"Processing: {file_path}")
            extract_responses(str(file_path))


def analyze_response_frequencies(
    experiments_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze frequencies and ratios of extracted response letters across all experiment files
    Returns:
        tuple: (frequency_df, ratio_df) where
            - frequency_df: DataFrame with raw counts
            - ratio_df: DataFrame with proportions of total responses
    """
    experiments_dir = Path(experiments_dir)
    freq_data = {}
    ratio_data = {}

    # Scan for extracted response files
    for model_dir in experiments_dir.iterdir():
        if not model_dir.is_dir():
            continue

        extracted_dir = model_dir / "extracted_responses"
        if not extracted_dir.exists():
            continue

        for file_path in extracted_dir.glob("*_extracted_responses.jsonl"):
            key = (model_dir.name, file_path.name)
            letter_counts = {"A": 0, "B": 0, "C": 0, "D": 0, "None": 0}
            total_responses = 0

            # Process each response in the file
            with open(file_path, "r") as f:
                for line in f:
                    data = json.loads(line)
                    letter = data["extracted_response_letter"]
                    letter = str(letter) if letter is not None else "None"
                    letter_counts[letter] += 1
                    total_responses += 1

            # Store raw frequencies
            freq_data[key] = {**letter_counts, "total_responses": total_responses}

            # Calculate and store ratios
            ratio_data[key] = {
                letter: count / total_responses
                for letter, count in letter_counts.items()
            }
            ratio_data[key]["total_responses"] = total_responses

    # Convert to DataFrames
    freq_df = pd.DataFrame.from_dict(freq_data, orient="index")
    ratio_df = pd.DataFrame.from_dict(ratio_data, orient="index")

    # Reset index for both DataFrames
    for df in [freq_df, ratio_df]:
        df.reset_index(inplace=True)
        df.rename(
            columns={"level_0": "model_name", "level_1": "file_name"}, inplace=True
        )

    return freq_df, ratio_df


def analyze_ground_truth_frequencies(
    experiments_dir: str | Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Analyze frequencies and ratios of ground truth letters across all experiment files
    Returns:
        tuple: (frequency_df, ratio_df) where
            - frequency_df: DataFrame with raw counts
            - ratio_df: DataFrame with proportions of total responses
    """
    experiments_dir = Path(experiments_dir)
    freq_data = {}
    ratio_data = {}

    # Scan for extracted response files
    for model_dir in experiments_dir.iterdir():
        if not model_dir.is_dir():
            continue

        extracted_dir = model_dir / "extracted_responses"
        if not extracted_dir.exists():
            continue

        for file_path in extracted_dir.glob("*_extracted_responses.jsonl"):
            key = (model_dir.name, file_path.name)
            letter_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
            total_responses = 0

            # Process each response in the file
            with open(file_path, "r") as f:
                for line in f:
                    data = json.loads(line)
                    # Extract first letter from ground truth
                    ground_truth = data["ground_truth"].strip().upper()
                    letter = ground_truth[0] if ground_truth else None
                    if letter and (letter in letter_counts):
                        letter_counts[letter] += 1
                        total_responses += 1

            # Store raw frequencies
            freq_data[key] = {**letter_counts, "total_responses": total_responses}

            # Calculate and store ratios
            ratio_data[key] = {
                letter: count / total_responses
                for letter, count in letter_counts.items()
            }
            ratio_data[key]["total_responses"] = total_responses

    # Convert to DataFrames
    freq_df = pd.DataFrame.from_dict(freq_data, orient="index")
    ratio_df = pd.DataFrame.from_dict(ratio_data, orient="index")

    # Reset index for both DataFrames
    for df in [freq_df, ratio_df]:
        df.reset_index(inplace=True)
        df.rename(
            columns={"level_0": "model_name", "level_1": "file_name"}, inplace=True
        )

    return freq_df, ratio_df


if __name__ == "__main__":
    experiments_dir = (
        "all_experiments/table1_main_performance/ai_responses/question_bank_1000"
    )

    # Extract responses if not already done
    extract_all_responses(experiments_dir)

    # Analyze frequencies and ratios
    freq_df, ratio_df = analyze_response_frequencies(experiments_dir)

    print("\nResponse Letter Frequencies (raw counts):")
    print(freq_df)
    print("\nResponse Letter Ratios:")
    print(ratio_df)

    # Save to CSV
    freq_df.to_csv(
        "outputs/benchmarking/reports/response_stats/response_frequencies.csv",
        index=False,
    )
    ratio_df.to_csv(
        "outputs/benchmarking/reports/response_stats/response_ratios.csv", index=False
    )

    # Analyze ground truth frequencies and ratios

    # gt_freq_df, gt_ratio_df = analyze_ground_truth_frequencies(experiments_dir)

    # print("\nGround Truth Letter Frequencies (raw counts):")
    # print(gt_freq_df)
    # print("\nGround Truth Letter Ratios:")
    # print(gt_ratio_df)

    # # Save to CSV
    # gt_freq_df.to_csv(
    #     "outputs/benchmarking/reports/response_stats/ground_truth_frequencies.csv",
    #     index=False,
    # )
    # gt_ratio_df.to_csv(
    #     "outputs/benchmarking/reports/response_stats/ground_truth_ratios.csv",
    #     index=False,
    # )
