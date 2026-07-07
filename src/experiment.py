import csv
import os
from tqdm import tqdm

from src.data.distortion import create_noisy_context, generate_wrong_answer
from src.data.prompting import build_prompt, build_adversarial_prompt
from src.data.verification import generate_with_verification
from src.evaluation.metrics import is_correct, classify_verification_effect


class ExperimentRunner:
    """
    Runs baseline / noisy / adversarial conditions, each with a
    verification follow-up, across a set of HotpotQA examples. Writes
    one row per (example, condition) to CSV as it goes -- a crash
    partway through a long run won't lose everything already computed.
    """

    FIELDNAMES = [
        "example_id", "question", "gold_answer", "condition",
        "first_answer", "first_correct",
        "final_answer", "final_correct",
        "verification_effect",
    ]

    def __init__(self, client, output_path="../results/experiment_results.csv"):
        self.client = client
        self.output_path = output_path
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDNAMES).writeheader()

    def _run_condition(self, example, condition_name, prompt):
        first_answer, final_answer = generate_with_verification(self.client, prompt)
        first_correct = is_correct(first_answer, example["answer"])
        final_correct = is_correct(final_answer, example["answer"])
        row = {
            "example_id": example["id"], "question": example["question"],
            "gold_answer": example["answer"], "condition": condition_name,
            "first_answer": first_answer, "first_correct": first_correct,
            "final_answer": final_answer, "final_correct": final_correct,
            "verification_effect": classify_verification_effect(first_correct, final_correct),
        }
        with open(self.output_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDNAMES).writerow(row)

    def run(self, dataset):
        for example in tqdm(dataset, desc="Running experiment"):
            try:
                self._run_condition(example, "baseline", build_prompt(example))

                noisy_context, *_ = create_noisy_context(self.client, example, fact_index=0)
                self._run_condition(example, "noisy", build_prompt(example, context=noisy_context))

                wrong_answer = generate_wrong_answer(self.client, example["question"], example["answer"])
                self._run_condition(example, "adversarial", build_adversarial_prompt(example, wrong_answer))
            except Exception as e:
                print(f"  [skipped example {example.get('id', '?')}: {e}]")
                continue
