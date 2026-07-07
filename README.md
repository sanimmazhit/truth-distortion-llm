# Truth, Lies, and Reasoning Machines

Investigates how an LLM's factual accuracy and self-consistency degrade
under corrupted evidence (noisy) and social pressure (adversarial),
and whether self-verification prompting ("are you sure?") mitigates
this degradation.

## Research Question
To what extent does exposure to noisy (contradictory) and adversarial
(misleading) context degrade an LLM's factual accuracy and internal
consistency on multi-hop question answering, and does explicit
self-verification prompting mitigate this degradation?

## Setup
- Model: `llama3.1:8b` via [Ollama](https://ollama.com) (local inference)
- Dataset: [HotpotQA](https://hotpotqa.github.io/), distractor setting,
  50-example random sample (seed=42)

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.1:8b
```

## Repository Structure
- `src/models/` — Ollama client wrapper
- `src/data/` — dataset loading, distortion generation, prompt building, verification
- `src/evaluation/` — accuracy and self-contradiction metrics
- `src/experiment.py` — experiment runner
- `notebooks/` — demonstration and analysis (run in order: `01_test_connection.ipynb`, `02_explore_data.ipynb`)
- `results/` — experiment CSVs and figures

## Reproducing Results
Run all cells in `notebooks/02_explore_data.ipynb` in order. The full
50-example run takes approximately 1.5-2 hours on a CPU-only laptop.

## Key Findings
- Corrupted evidence (noisy) degrades accuracy far more than social
  pressure alone (adversarial): -18 points vs. essentially no change.
- Self-verification prompting has an uneven effect: it improves
  accuracy only in the noisy condition (+4 points net), while
  *reducing* accuracy in baseline (-10) and adversarial (-8) conditions.
- Adversarial pressure produces the highest self-contradiction rate
  (0.14) despite barely affecting final accuracy -- social pressure
  destabilizes the model's stated reasoning more than its answers.

## AI Usage Disclaimer
Parts of this project (code structure, debugging, drafting of
descriptive text) were developed with the assistance of Claude
(Anthropic). All content was reviewed, tested, and validated by me.
I take full responsibility for the final content, its accuracy, and
its academic integrity.