import argparse
import importlib
from pathlib import Path
import importlib.util
from typing import Type, List, Optional

from dvbench.inference.models.model_test import test_model_with_questions
from dvbench.inference.models.index import SUPPORTED_MODELS
from dvbench.inference.models.base import ModelConfig

from dvbench.benchmarking.evaluate import (
    evaluate_model,
    DomainKnowledgePlacement,
)


def load_config(config_path: str | Path, model_class: Type) -> ModelConfig:
    """
    Dynamically load the config from the specified path with validation rules:
    1. Class name must end with 'ModelConfig'
    2. Class must be a subclass of ModelConfig but not ModelConfig itself
    3. Class name must start with the model's class name
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    model_name = model_class.__name__

    # Use importlib.import_module() when loading a module using its standard Python import path.
    # Use importlib.util.spec_from_file_location() and module_from_spec() when loading a module from an arbitrary file path outside of the standard Python import paths.

    # Import the module from the full package path
    module_path = str(config_path).replace("/", ".").replace(".py", "")
    if module_path.startswith("."):
        module_path = module_path[1:]
    try:
        module = importlib.import_module(module_path)
    except ImportError:
        # Fallback to spec_from_file_location if direct import fails
        spec = importlib.util.spec_from_file_location(config_path.stem, config_path)
        if spec is None or spec.loader is None:
            raise ImportError(
                f"Could not load module spec or loader from {config_path}"
            )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

    # Find valid config classes with all conditions
    config_classes = [
        obj
        for name, obj in module.__dict__.items()
        if (
            name.endswith("Config")  # Must end with ModelConfig
            and name.upper().startswith(
                model_name.upper()
            )  # Must start with model class name
            and isinstance(obj, type)  # Must be a class
            and issubclass(obj, ModelConfig)  # Must be a subclass of ModelConfig
            and obj != ModelConfig  # Must not be ModelConfig itself
        )
    ]

    if not config_classes:
        raise ValueError(
            f"No valid ModelConfig class found in {config_path}. "
            f"Class must: \n"
            f"1. End with 'ModelConfig'\n"
            f"2. Be a subclass of ModelConfig (but not ModelConfig itself)\n"
            f"3. Start with '{model_name}'"
        )

    if len(config_classes) > 1:
        raise ValueError(
            f"Multiple valid ModelConfig classes found in {config_path}: {[c.__name__ for c in config_classes]}"
        )

    config_instance = config_classes[0]()

    # Add debug prints
    print("\nDebug load_config:")
    print(f"Config class: {config_classes[0]}")
    print(f"Config instance type: {type(config_instance)}")
    print(f"Config instance: {config_instance}")
    print(f"Config instance dict: {config_instance.__dict__}")

    return config_instance


def main():
    print(f"SUPPORTED_MODELS: {SUPPORTED_MODELS}")

    parser = argparse.ArgumentParser(
        description="Run model evaluation with specified configuration"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(SUPPORTED_MODELS.keys()),
        help="Name of the model to evaluate",
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the configuration file"
    )
    parser.add_argument(
        "--question_path",
        type=str,
        required=True,
        help="Path to the JSONL file containing questions",
    )
    parser.add_argument(
        "--output_dir", type=str, help="Directory to save AI responses", default=None
    )
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Batch size for processing questions"
    )
    parser.add_argument(
        "--start_pos", type=int, default=0, help="Starting position in question bank"
    )
    parser.add_argument(
        "--end_pos",
        type=int,
        default=-1,
        help="Ending position in question bank (-1 for all questions)",
    )
    parser.add_argument(
        "--placement",
        choices=[
            "none",
            "all",
            "all_and_none",
            "before_question",
            "after_question",
            "after_options",
        ],
        default="none",
        help="Knowledge placement strategy (all, none, or specific placement)",
    )

    args = parser.parse_args()

    # Check if model exists in supported models
    if args.model not in SUPPORTED_MODELS:
        raise ValueError(
            f"Model '{args.model}' not found in supported models: {list(SUPPORTED_MODELS.keys())}"
        )
    model_name = args.model
    print(f"Model name: {model_name}")
    model_class = SUPPORTED_MODELS[model_name]

    # Simplified placement handling
    if args.placement == "all_and_none":
        placement: List[Optional[DomainKnowledgePlacement]] = list(
            DomainKnowledgePlacement
        ) + [None]
    elif args.placement == "none":
        placement = [None]
    elif args.placement == "all":
        placement = [
            DomainKnowledgePlacement.BEFORE_QUESTION,
            DomainKnowledgePlacement.AFTER_QUESTION,
            DomainKnowledgePlacement.AFTER_OPTIONS,
        ]
    else:
        placement = [DomainKnowledgePlacement[args.placement.upper()]]

    # Determine output directory
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(args.question_path).parent.parent / "ai_responses"
    )

    try:
        # Load config
        configs = load_config(args.config, model_class)
        print("\nDebug main:")
        print(f"Configs type: {type(configs)}")
        print(f"Configs: {configs}")
        print(f"Configs dict: {configs.__dict__}")

        # Initialize model
        model = model_class(configs=configs)

        # Run evaluation
        # test_model_with_questions(
        #     model=model,
        #     question_bank_path=args.question_path
        # )
        evaluate_model(
            model=model,
            model_name=model_name,
            question_bank_path=args.question_path,
            ai_response_output_dir=output_dir,
            domain_knowledge_placement=placement,
            batch_size=args.batch_size,
            start_pos=args.start_pos,
            end_pos=args.end_pos,
        )

    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise


if __name__ == "__main__":
    # import sys

    # sys.argv = [
    #     sys.argv[0],
    #     "--model",
    #     "pllava_34b",
    #     "--config",
    #     "/home/username/Projects/DrivingVllmBench/all_experiments/table1_main_performance/model_configs/hf/config_pllava_34b.py",
    #     "--question_path",
    #     "/home/username/Projects/DrivingVllmBench/all_experiments/table1_main_performance/question_bank/question_bank_1000.jsonl",
    #     "--batch_size",
    #     "1",
    #     "--placement",
    #     "none",
    #     "--output_dir",
    #     "/home/username/Projects/DrivingVllmBench/all_experiments/table1_main_performance/ai_responses",
    # ]

    main()
