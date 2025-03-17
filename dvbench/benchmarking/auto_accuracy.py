from pathlib import Path
import pandas as pd
from typing import List, Dict, Union, Any
from dvbench.benchmarking.calc_accuracy import AccuracyEvaluator
import json


class AutoAccuracyEvaluator:
    def __init__(self, experiments_dir: Union[str, Path]):
        self.experiments_dir = Path(experiments_dir)
        self.results_cache = {}  # Cache for storing evaluation results

    def scan_experiment_files(self) -> List[Dict[str, Union[str, Path]]]:
        """
        Scan the experiments directory for AI response files
        Returns a list of dicts containing model_name and file_path
        """
        experiment_files = []

        # Only look at direct children of experiments_dir
        for model_dir in self.experiments_dir.iterdir():
            if not model_dir.is_dir():
                continue

            for file_path in model_dir.glob("ai_responses_*.jsonl"):
                experiment_files.append(
                    {"model_name": model_dir.name, "file_path": file_path}
                )

        return experiment_files

    def evaluate_single_file(
        self, file_info: Dict[str, Union[str, Path]], strategy: str = "individual"
    ) -> Dict[str, Any]:
        """
        Evaluate a single experiment file using specified strategy
        """
        evaluator = AccuracyEvaluator(str(file_info["file_path"]))

        # Create output paths for results
        output_dir = Path(file_info["file_path"]).parent / "eval_results"
        output_dir.mkdir(parents=True, exist_ok=True)

        file_stem = Path(file_info["file_path"]).stem
        individual_path = output_dir / f"{file_stem}_individual_results.jsonl"
        group_path = output_dir / f"{file_stem}_group_results.jsonl"

        # Get individual results
        individual_results = evaluator.evaluate_individual(str(individual_path))

        # Get group results if needed
        group_results = None
        if strategy == "group":
            group_results = evaluator.evaluate_group(
                individual_results, str(group_path)
            )

        # Calculate hierarchical accuracies
        results = individual_results if strategy == "individual" else group_results
        if results is None:
            raise ValueError(f"No results available for strategy: {strategy}")

        use_group = strategy == "group"

        return {
            "model_name": file_info["model_name"],
            "file_name": Path(file_info["file_path"]).name,
            "level0": evaluator.calc_level_zero_accuracy(
                results, use_group_accuracy=use_group
            ),
            "level1": evaluator.calc_level_one_accuracy(
                results, use_group_accuracy=use_group
            ),
            "level2": evaluator.calc_level_two_accuracy(
                results, use_group_accuracy=use_group
            ),
            "level3": evaluator.calc_level_three_accuracy(
                results, use_group_accuracy=use_group
            ),
        }

    def evaluate_all(self, strategy: str = "individual") -> None:
        """
        Evaluate all experiment files and cache results
        """
        experiment_files = self.scan_experiment_files()
        self.results_cache[strategy] = []

        for file_info in experiment_files:
            result = self.evaluate_single_file(file_info, strategy)
            self.results_cache[strategy].append(result)

    def _convert_to_dataframe(
        self, level_results: List[Dict], level: int
    ) -> pd.DataFrame:
        """
        Convert hierarchical results to pandas DataFrame with models as rows
        and categories as columns
        """
        # Create a dictionary to store model results
        model_data = {}

        for result in level_results:
            model_name = result["model_name"]
            file_name = result["file_name"]
            accuracies = result[f"level{level}"][model_name]

            key = (model_name, file_name)  # Use tuple as key
            if key not in model_data:
                model_data[key] = {}

            # For each level, store accuracies differently
            if level == 0:
                # Direct mapping of category to accuracy
                model_data[key].update(accuracies)
            elif level == 1:
                # Direct mapping of category to accuracy
                model_data[key].update(accuracies)
            else:
                # Split hierarchical paths and use last component as column name
                for category, accuracy in accuracies.items():
                    category_parts = category.split("/")
                    column_name = category_parts[-1]  # Use last part as column
                    model_data[key][column_name] = accuracy

        # Convert to DataFrame with models as index
        df = pd.DataFrame.from_dict(model_data, orient="index")

        # Reset index and rename columns
        df.reset_index(inplace=True)
        df.rename(
            columns={"level_0": "model_name", "level_1": "file_name"}, inplace=True
        )

        return df

    def get_level_zero_accuracy(self, strategy: str = "individual") -> pd.DataFrame:
        """Get overall accuracy results as DataFrame"""
        if strategy not in self.results_cache:
            self.evaluate_all(strategy)
        return self._convert_to_dataframe(self.results_cache[strategy], 0)

    def get_level_one_accuracy(self, strategy: str = "individual") -> pd.DataFrame:
        """Get level 1 accuracy results as DataFrame"""
        if strategy not in self.results_cache:
            self.evaluate_all(strategy)
        return self._convert_to_dataframe(self.results_cache[strategy], 1)

    def get_level_two_accuracy(self, strategy: str = "individual") -> pd.DataFrame:
        """Get level 2 accuracy results as DataFrame"""
        if strategy not in self.results_cache:
            self.evaluate_all(strategy)
        return self._convert_to_dataframe(self.results_cache[strategy], 2)

    def get_level_three_accuracy(self, strategy: str = "individual") -> pd.DataFrame:
        """Get level 3 accuracy results as DataFrame"""
        if strategy not in self.results_cache:
            self.evaluate_all(strategy)
        return self._convert_to_dataframe(self.results_cache[strategy], 3)

    def save_results(self, output_dir: Union[str, Path]) -> None:
        """
        Save all results to CSV files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for strategy in self.results_cache:
            strategy_dir = output_dir / strategy
            strategy_dir.mkdir(exist_ok=True)

            # Save each level's results
            self.get_level_zero_accuracy(strategy).to_csv(
                strategy_dir / "level0_accuracy.csv", index=False
            )
            self.get_level_one_accuracy(strategy).to_csv(
                strategy_dir / "level1_accuracy.csv", index=False
            )
            self.get_level_two_accuracy(strategy).to_csv(
                strategy_dir / "level2_accuracy.csv", index=False
            )
            self.get_level_three_accuracy(strategy).to_csv(
                strategy_dir / "level3_accuracy.csv", index=False
            )


def main():
    # Example usage
    experiments_dir = Path(
        "/home/username/Projects/DrivingVllmBench/all_experiments/table1_main_performance/ai_responses/question_bank_1000"
    )
    evaluator = AutoAccuracyEvaluator(experiments_dir)

    # Evaluate using both strategies
    for strategy in ["individual", "group"]:
        evaluator.evaluate_all(strategy)

    # Get results as DataFrames
    level0_df = evaluator.get_level_zero_accuracy("individual")
    level1_df = evaluator.get_level_one_accuracy("individual")
    level2_df = evaluator.get_level_two_accuracy("individual")
    level3_df = evaluator.get_level_three_accuracy("individual")

    # Save results
    evaluator.save_results("accuracy_results")

    # Print summary
    print("\nLevel 0 (Overall) Accuracy Summary:")
    print(level0_df.groupby("model_name")["Overall"].mean())

    print("\nLevel 1 Accuracy Summary:")
    print(level1_df.groupby("model_name")["accuracy"].mean())

    print("\nLevel 2 Accuracy Summary:")
    print(level2_df.groupby(["level1", "level2"])["accuracy"].mean())

    print("\nLevel 3 Accuracy Summary:")
    print(level3_df.groupby(["level1", "level2", "level3"])["accuracy"].mean())


if __name__ == "__main__":
    main()
