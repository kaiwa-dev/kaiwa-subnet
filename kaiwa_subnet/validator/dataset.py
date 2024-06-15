import time

from datasets import load_dataset
import random

random.seed(int(time.time()))


class ValidationDataset:
    def __init__(self) -> None:
        self.dataset = load_dataset("chenghao/quora_questions")

    def random_prompt(self) -> str:
        index = random.randint(0, len(self.dataset["train"]))
        return self.dataset["train"][index]["questions"]


if __name__ == "__main__":
    ValidationDataset()
