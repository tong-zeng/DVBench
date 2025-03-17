from pathlib import Path
from typing import List, Dict, Any, Union
from collections import defaultdict
from dvbench.question_bank_gen.gen_question_bank import (
    save_to_jsonl,
    load_from_jsonl,
)
import re

category_mapping = {
    "Perception": {
        "Environmental Conditions": {
            "Atmospheric Conditions": ["Light", "Weather"],
            "Road Surface Conditions": ["Surface_Cndtn"],
        },
        "Physical Infrastructure": {
            "Road Geometry": ["Rd_Alignment", "Grade"],
            "Lane Configuration": ["Contiguous_Trvl_Lanes", "Thru_Trvl_Lanes"],
        },
        "Operational Constraints": {
            "Traffic Conditions": ["Traffic_Flow", "Traffic_Density"],
            "Traffic Control Devices": ["Traffic_Control"],
            "Visibility Constraints": ["Vis_Obstructions"],
        },
        "Objects": {
            "Static Objects": ["Objects_Animals", "Object_2_Type"],
            "Other Road Users": [
                "Others_Involved",
                "Non_Motorist_2_Pre_Incident_Maneuver",
                "Non_Motorist_2_Evasive_Maneuver",
            ],
        },
        "Zones": {
            "Geographic Zones": ["Locality"],
            "Intersection and Junctions": [
                "Relation_to_Junction",
                "Intersection_Influence",
            ],
            "Construction Zones": ["Construction_Zone"],
        },
    },
    "Reasoning": {
        "Event Understanding": {
            "Event Characteristics": ["Event_Nature_1", "Incident_Type_1"],
            "Event Triggers": ["Precipitating_Event"],
            "Event Severity": ["Event_Severity_1", "Crash_Severity_1"],
        },
        "Behavior & Maneuver Analysis": {
            "Vehicle Maneuvers": [
                "Pre_Incident_Maneuver",
                "V1_Evasive_Maneuver_1",
                "V1_Post_Maneuver_Control_1",
            ],
            "Maneuver Evaluation": ["Maneuver_Judgment"],
        },
        "Spatial Reasoning": {
            "Road and Terrain Features": ["Rd_Alignment", "Grade", "Locality"],
            "Lane Positioning": [
                "V1_Lane_Occupied",
                "Contiguous_Tr vl_Lanes",
                "Thru_Trvl_Lanes",
            ],
            "Spatial Relationships": [
                "Relation_to_Junction",
                "Intersection_Influence",
                "Object_2_Location",
            ],
        },
        "Risk & Hazard Assessment": {
            "Environmental Hazards": [
                "Vis_Obstructions",
                "Surface_Cndtn",
                "Weather",
                "Light",
            ],
            "Traffic Hazards": ["Traffic_Flow", "Traffic_Density", "Traffic_Control"],
            "Obstruction Hazards": [
                "Objects_Animals",
                "Construction_Zone",
                "Others_Involved",
            ],
        },
        "Causal and Responsibility": {
            "Cause and Effect": [
                "Precipitating_Event",
                "Pre_Incident_Maneuver",
                "Maneuver_Judgment",
            ],
            "Fault Analysis": ["Fault", "Others_Involved"],
        },
    },
}


class AccuracyEvaluator:
    def __init__(self, input_jsonl_path: str):
        self.input_path = input_jsonl_path
        self.data = load_from_jsonl(input_jsonl_path)
        self.model_name = Path(input_jsonl_path).parent.name
        self.answer_pattern = re.compile(r"[\s][ABCD]\.\s*")

    def is_response_correct(self, response: str, ground_truth: str) -> bool:
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
        matches = self.answer_pattern.findall(response)
        if len(matches) == 1:
            matched_answer = matches[0].strip()
            if ground_truth.startswith(matched_answer):
                print(
                    f"============================\nMatch: {matched_answer}, original: {response}"
                )
                return True
        return False

    def evaluate_individual(self, output_path: str) -> List[Dict[str, Any]]:
        """Evaluate each response individually"""
        results = []
        for item in self.data:
            result = {
                "question_field": item["question_field"],
                "video_id": item["video_id"],
                "ground_truth": item["ground_truth"],
                "response": item["response"],
                "is_correct": self.is_response_correct(
                    item["response"], item["ground_truth"]
                ),
            }
            results.append(result)

        save_to_jsonl(results, output_path)
        return results

    def evaluate_group(
        self, individual_results: List[Dict[str, Any]], output_path: str
    ) -> List[Dict[str, Any]]:
        """Evaluate responses grouped by question_field and video_id"""
        groups = defaultdict(list)
        for result in individual_results:
            key = (result["question_field"], result["video_id"])
            groups[key].append(result)

        group_results = []
        for (field, vid), items in groups.items():
            group_result = {
                "question_field": field,
                "video_id": vid,
                "group_size": len(items),
                "correct_count": sum(item["is_correct"] for item in items),
                "correct_ratio": sum(item["is_correct"] for item in items) / len(items),
                "is_correct": all(item["is_correct"] for item in items),
            }
            group_results.append(group_result)

        save_to_jsonl(group_results, output_path)
        return group_results

    def aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Aggregate results by question_field and calculate accuracy"""
        field_results = defaultdict(list)
        for result in results:
            field_results[result["question_field"]].append(result["is_correct"])

        accuracies = {}
        for field, correctness in field_results.items():
            accuracy = sum(correctness) / len(correctness)
            accuracies[field] = accuracy

        return accuracies

    def report_performance(self, individual_path: str, group_path: str) -> str:
        """Generate performance report as markdown table"""
        individual_results = load_from_jsonl(individual_path)
        group_results = load_from_jsonl(group_path)

        # Add check for empty results
        if not individual_results or not group_results:
            return (
                "No results found. Please check if input files exist and contain data."
            )

        individual_acc = self.aggregate_results(individual_results)
        group_acc = self.aggregate_results(group_results)

        # Calculate overall accuracy
        overall_individual = sum(
            result["is_correct"] for result in individual_results
        ) / len(individual_results)
        overall_group = sum(result["is_correct"] for result in group_results) / len(
            group_results
        )

        # Generate markdown table
        table = "| Question Field | Individual Accuracy | Group Accuracy |\n"
        table += "|---------------|-------------------|---------------|\n"

        for field in sorted(individual_acc.keys()):
            table += (
                f"| {field} | {individual_acc[field]:.2%} | {group_acc[field]:.2%} |\n"
            )

        # Add overall row
        table += f"| Overall | {overall_individual:.2%} | {overall_group:.2%} |\n"

        return table

    def _get_field_to_hierarchy_mapping(self) -> Dict[str, List[tuple]]:
        """Create mapping from question_field to its hierarchy paths"""
        field_mapping = defaultdict(list)

        for level1, level1_data in category_mapping.items():
            for level2, level2_data in level1_data.items():
                for level3, fields in level2_data.items():
                    for field in fields:
                        field_mapping[field].append((level1, level2, level3))

        return dict(field_mapping)

    def _calculate_hierarchical_accuracy(
        self, results: List[Dict[str, Any]], use_group_accuracy: bool = False
    ) -> Dict[str, Dict[str, Dict[str, Dict[str, List[bool]]]]]:
        """Calculate accuracy for each level of hierarchy, including level zero"""
        field_mapping = self._get_field_to_hierarchy_mapping()

        # Initialize nested defaultdict for storing results
        hierarchical_results = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        )

        # Add virtual root node "Overall" at level zero
        root_node = "Overall"

        # Collect results by hierarchy
        for result in results:
            field = result["question_field"]
            is_correct = bool(result["is_correct"])  # Ensure boolean type

            # Add to overall results
            hierarchical_results[root_node]["All"]["All"]["All"].append(is_correct)

            if field in field_mapping:
                # A field can belong to multiple categories
                for level1, level2, level3 in field_mapping[field]:
                    hierarchical_results[level1][level2][level3]["All"].append(
                        is_correct
                    )

        return dict(
            {
                k: dict(
                    {
                        k2: dict({k3: dict(v3) for k3, v3 in v2.items()})
                        for k2, v2 in v.items()
                    }
                )
                for k, v in hierarchical_results.items()
            }
        )

    def calc_level_zero_accuracy(
        self,
        results: List[Dict[str, Any]],
        use_group_accuracy: bool = False,
        return_format: str = "json",
    ) -> Union[str, Dict[str, float]]:
        """Report overall accuracy across all categories"""
        hierarchical_results = self._calculate_hierarchical_accuracy(
            results, use_group_accuracy
        )

        if return_format == "markdown":
            table = "| Overall Accuracy |\n"
            table += "|------------------|\n"

            scores = hierarchical_results["Overall"]["All"]["All"]["All"]
            accuracy = sum(scores) / len(scores)
            table += f"| {accuracy:.2%} |\n"
            return table

        # JSON format
        scores = hierarchical_results["Overall"]["All"]["All"]["All"]
        accuracy = sum(scores) / len(scores)
        return {self.model_name: {"Overall": accuracy}}

    def calc_level_one_accuracy(
        self,
        results: List[Dict[str, Any]],
        use_group_accuracy: bool = False,
        return_format: str = "json",
    ) -> Union[str, Dict[str, Dict[str, float]]]:
        """Report accuracy for top level categories (Perception and Reasoning)"""
        hierarchical_results = self._calculate_hierarchical_accuracy(
            results, use_group_accuracy
        )

        if return_format == "markdown":
            table = "| Level 1 | Accuracy |\n"
            table += "|----------|----------|\n"
            for level1 in sorted(hierarchical_results.keys()):
                if level1 == "Overall":  # Skip the overall node
                    continue
                all_scores = []
                for level2_data in hierarchical_results[level1].values():
                    for level3_data in level2_data.values():
                        for scores in level3_data.values():
                            all_scores.extend(scores)
                accuracy = sum(all_scores) / len(all_scores)
                table += f"| {level1} | {accuracy:.2%} |\n"
            return table

        # JSON format
        accuracies = {}
        for level1 in sorted(hierarchical_results.keys()):
            if level1 == "Overall":  # Skip the overall node
                continue
            all_scores = []
            for level2_data in hierarchical_results[level1].values():
                for level3_data in level2_data.values():
                    for scores in level3_data.values():
                        all_scores.extend(scores)
            accuracy = sum(all_scores) / len(all_scores)
            accuracies[level1] = accuracy

        return {self.model_name: accuracies}

    def calc_level_two_accuracy(
        self,
        results: List[Dict[str, Any]],
        use_group_accuracy: bool = False,
        return_format: str = "json",
    ) -> Union[str, Dict[str, Dict[str, float]]]:
        """Report accuracy for each level 2 category"""
        hierarchical_results = self._calculate_hierarchical_accuracy(
            results, use_group_accuracy
        )

        if return_format == "markdown":
            table = "| Level 1 | Level 2 | Accuracy |\n"
            table += "|----------|----------|----------|\n"
            for level1 in sorted(hierarchical_results.keys()):
                if level1 == "Overall":  # Skip the overall node
                    continue
                for level2 in sorted(hierarchical_results[level1].keys()):
                    all_scores = []
                    for level3_data in hierarchical_results[level1][level2].values():
                        for scores in level3_data.values():
                            all_scores.extend([bool(score) for score in scores])

                    if all_scores:  # Check if we have any scores
                        accuracy = sum(all_scores) / len(all_scores)
                        table += f"| {level1} | {level2} | {accuracy:.2%} |\n"
            return table

        # JSON format
        accuracies = {}
        for level1 in sorted(hierarchical_results.keys()):
            if level1 == "Overall":  # Skip the overall node
                continue
            for level2 in sorted(hierarchical_results[level1].keys()):
                all_scores = []
                for level3_data in hierarchical_results[level1][level2].values():
                    for scores in level3_data.values():
                        all_scores.extend([bool(score) for score in scores])

                if all_scores:  # Check if we have any scores
                    accuracy = sum(all_scores) / len(all_scores)
                    accuracies[f"{level1}/{level2}"] = accuracy

        return {self.model_name: accuracies}

    def calc_level_three_accuracy(
        self,
        results: List[Dict[str, Any]],
        use_group_accuracy: bool = False,
        return_format: str = "json",
    ) -> Union[str, Dict[str, Dict[str, float]]]:
        """Report accuracy for each level 3 category"""
        hierarchical_results = self._calculate_hierarchical_accuracy(
            results, use_group_accuracy
        )

        if return_format == "markdown":
            table = "| Level 1 | Level 2 | Level 3 | Accuracy |\n"
            table += "|----------|----------|----------|----------|\n"
            for level1 in sorted(hierarchical_results.keys()):
                if level1 == "Overall":  # Skip the overall node
                    continue
                for level2 in sorted(hierarchical_results[level1].keys()):
                    for level3 in sorted(hierarchical_results[level1][level2].keys()):
                        scores = hierarchical_results[level1][level2][level3]["All"]
                        # Ensure all scores are boolean
                        scores = [bool(score) for score in scores]
                        if scores:  # Check if we have any scores
                            accuracy = sum(scores) / len(scores)
                            table += (
                                f"| {level1} | {level2} | {level3} | {accuracy:.2%} |\n"
                            )
            return table

        # JSON format
        accuracies = {}
        for level1 in sorted(hierarchical_results.keys()):
            if level1 == "Overall":  # Skip the overall node
                continue
            for level2 in sorted(hierarchical_results[level1].keys()):
                for level3 in sorted(hierarchical_results[level1][level2].keys()):
                    scores = hierarchical_results[level1][level2][level3]["All"]
                    # Ensure all scores are boolean
                    scores = [bool(score) for score in scores]
                    if scores:  # Check if we have any scores
                        accuracy = sum(scores) / len(scores)
                        accuracies[f"{level1}/{level2}/{level3}"] = accuracy

        return {self.model_name: accuracies}


def main():
    # input_ai_response = "outputs/benchmarking/ai_responses/question_bank_100/ai_responses_question_bank_100_0_102_model_MiniCPMVvLLM_domain_knowledge_placement_after_options.jsonl"
    input_ai_response = "/home/username/Projects/DrivingVllmBench/all_experiments/table1_main_performance/ai_responses/question_bank_1000/qwen2_vl_7b/ai_responses_question_bank_1000_0_all_model_qwen2_vl_7b_domain_knowledge_none.jsonl"

    input_ai_response_path = Path(input_ai_response)
    input_file_name = input_ai_response_path.stem

    eval_output_dir = input_ai_response_path.parent / "eval_results"
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    individual_results_path = str(
        (eval_output_dir / f"{input_file_name}_individual_results.jsonl").resolve()
    )
    group_results_path = str(
        (eval_output_dir / f"{input_file_name}_group_results.jsonl").resolve()
    )

    evaluator = AccuracyEvaluator(input_ai_response)

    # Evaluate individual responses
    individual_results = evaluator.evaluate_individual(individual_results_path)

    # Evaluate grouped responses
    group_results = evaluator.evaluate_group(individual_results, group_results_path)

    # # Generate and print performance report
    # report = evaluator.report_performance(individual_results_path, group_results_path)
    # print(report)

    # # Using individual results
    # level3_table = evaluator.calc_level_three_accuracy(
    #     individual_results, use_group_accuracy=False
    # )
    # level2_table = evaluator.calc_level_two_accuracy(
    #     individual_results, use_group_accuracy=False
    # )
    # level1_table = evaluator.calc_level_one_accuracy(
    #     individual_results, use_group_accuracy=False
    # )

    # print(level3_table)
    # print(level2_table)
    # print(level1_table)

    # Or using group results
    # level3_table = evaluator.calc_level_three_accuracy(
    #     group_results, use_group_accuracy=True
    # )
    # level2_table = evaluator.calc_level_two_accuracy(
    #     group_results, use_group_accuracy=True
    # )
    # level1_table = evaluator.calc_level_one_accuracy(
    #     group_results, use_group_accuracy=True
    # )

    # print(level3_table)
    # print(level2_table)
    # print(level1_table)

    # Calculate accuracies at all levels
    level0_results = evaluator.calc_level_zero_accuracy(
        group_results, use_group_accuracy=True, return_format="markdown"
    )
    level1_results = evaluator.calc_level_one_accuracy(
        group_results, use_group_accuracy=True, return_format="markdown"
    )
    level2_results = evaluator.calc_level_two_accuracy(
        group_results, use_group_accuracy=True, return_format="markdown"
    )
    level3_results = evaluator.calc_level_three_accuracy(
        group_results, use_group_accuracy=True, return_format="markdown"
    )

    print("\nOverall Accuracy:")
    print(level0_results)
    print("\nLevel 1 Accuracy:")
    print(level1_results)
    print("\nLevel 2 Accuracy:")
    print(level2_results)
    print("\nLevel 3 Accuracy:")
    print(level3_results)


if __name__ == "__main__":
    main()
