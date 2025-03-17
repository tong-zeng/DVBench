import argparse

from dvbench.inference.models.index import SUPPORTED_MODELS
from dvbench.inference.models.model_test import test_model_with_questions


parser = argparse.ArgumentParser()

parser.add_argument(
    "file", type=str, help="The path to the file containing the questions"
)
parser.add_argument(
    "--pos", type=int, help="The position of the questions to start and end"
)
args = parser.parse_args()

model = SUPPORTED_MODELS["mini_cpm_v"]()

test_model_with_questions(model)


print(args.file)
