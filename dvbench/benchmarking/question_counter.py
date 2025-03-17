from pathlib import Path
from typing import Dict, List, Any, Union
from collections import defaultdict
import json
from dvbench.benchmarking.calc_accuracy import category_mapping


class QuestionCounter:
    def __init__(self, input_jsonl_path: str | Path):
        self.input_path = Path(input_jsonl_path)
        self.data = self._load_data()
        self.category_mapping = category_mapping  # Using the existing mapping
        self.field_to_hierarchy = self._get_field_to_hierarchy_mapping()

    def _load_data(self) -> List[Dict[str, Any]]:
        """Load data from jsonl file"""
        data = []
        with open(self.input_path, "r") as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def _get_field_to_hierarchy_mapping(self) -> Dict[str, List[tuple]]:
        """Create mapping from question_field to its hierarchy paths"""
        field_mapping = defaultdict(list)

        for level1, level1_data in self.category_mapping.items():
            for level2, level2_data in level1_data.items():
                for level3, fields in level2_data.items():
                    for field in fields:
                        field_mapping[field].append((level1, level2, level3))

        return dict(field_mapping)

    def count_level_one(self) -> Dict[str, int]:
        """Count questions for each level 1 category"""
        counts = defaultdict(int)

        for item in self.data:
            field = item["question_field"]
            if field in self.field_to_hierarchy:
                for level1, _, _ in self.field_to_hierarchy[field]:
                    counts[level1] += 1

        return dict(counts)

    def count_level_two(self) -> Dict[str, int]:
        """Count questions for each level 2 category"""
        counts = defaultdict(int)

        for item in self.data:
            field = item["question_field"]
            if field in self.field_to_hierarchy:
                for level1, level2, _ in self.field_to_hierarchy[field]:
                    counts[f"{level1}/{level2}"] += 1

        return dict(counts)

    def count_level_three(self) -> Dict[str, int]:
        """Count questions for each level 3 category"""
        counts = defaultdict(int)

        for item in self.data:
            field = item["question_field"]
            if field in self.field_to_hierarchy:
                for level1, level2, level3 in self.field_to_hierarchy[field]:
                    counts[f"{level1}/{level2}/{level3}"] += 1

        return dict(counts)

    def report_counts(self, return_format: str = "markdown") -> str:
        """Generate count report in specified format"""
        l1_counts = self.count_level_one()
        l2_counts = self.count_level_two()
        l3_counts = self.count_level_three()

        if return_format == "markdown":
            report = "## Question Distribution\n\n"

            # Level 1
            report += "### Level 1 Categories\n"
            report += "| Category | Count |\n|----------|-------|\n"
            for cat, count in sorted(l1_counts.items()):
                report += f"| {cat} | {count} |\n"

            # Level 2
            report += "\n### Level 2 Categories\n"
            report += "| Category | Count |\n|----------|-------|\n"
            for cat, count in sorted(l2_counts.items()):
                report += f"| {cat} | {count} |\n"

            # Level 3
            report += "\n### Level 3 Categories\n"
            report += "| Category | Count |\n|----------|-------|\n"
            for cat, count in sorted(l3_counts.items()):
                report += f"| {cat} | {count} |\n"

            return report

        # JSON format
        return {"level1": l1_counts, "level2": l2_counts, "level3": l3_counts}


def main():
    input_path = "outputs/benchmarking/question_bank/question_bank_all.jsonl"
    counter = QuestionCounter(input_path)

    # Print markdown report
    print(counter.report_counts())

    # Or get individual level counts
    level1_counts = counter.count_level_one()
    level2_counts = counter.count_level_two()
    level3_counts = counter.count_level_three()

    print("\nLevel 1 counts:", level1_counts)
    print("\nLevel 2 counts:", level2_counts)
    print("\nLevel 3 counts:", level3_counts)


if __name__ == "__main__":
    main()
