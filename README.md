# Truth, Lies, and Reasoning Machines
*Testing LLM Robustness to Corrupted Evidence and Social Pressure in Multi-Hop QA*

NLP final project (P1) — Università degli Studi di Milano.

Investigates how an LLM's factual accuracy and self-consistency degrade
under corrupted evidence (noisy) and social pressure (adversarial),
and whether self-verification prompting ("are you sure?") mitigates
this degradation.

## Research Question
To what extent does exposure to noisy (contradictory) and adversarial
(misleading) context degrade an LLM's factual accuracy and internal
consistency on multi-hop question answering, and does explicit
self-verification prompting mitigate this degradation?

## Key Findings
- Corrupted evidence (noisy) degrades accuracy far more than social
  pressure alone (adversarial): first-pass accuracy drops 18 points
  (76% -> 58%) vs. essentially no change (76% -> 76%).
- Self-verification has an uneven effect: it improves accuracy only in
  the noisy condition (58% -> 62%, +4 net), while *reducing* accuracy
  in baseline (76% -> 66%, -10) and adversarial (76% -> 68%, -8).
- Adversarial pressure produces the highest self-contradiction rate
  (0.14) despite barely affecting final accuracy -- social pressure
  destabilizes the model's stated reasoning more than its answers.

## Pipeline Overview
load_hotpotqa()                              -> 50 HotpotQA examples (seed=42)
create_noisy_context() / generate_wrong_answer()  -> noisy / adversarial conditions
build_prompt() / build_adversarial_prompt()   -> the 3 condition prompts
generate_with_verification()                  -> first answer + post-verification answer
ExperimentRunner.run()                        -> results/experiment_results.csv (150 rows)
metrics.is_correct() / is_self_contradictory()-> accuracy + contradiction scores
02_explore_data.ipynb (scoring + plotting cells) -> results/*.png (3 figures)

## Repository Structure
.
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_test_connection.ipynb   # verifies Ollama connectivity
│   └── 02_explore_data.ipynb      # full pipeline: exploration -> smoke test -> 50-example run -> scoring -> figures
├── src/
│   ├── models/
│   │   └── ollama_client.py       # OllamaClient -- single-turn (.generate) and multi-turn (.chat) wrapper
│   ├── data/
│   │   ├── loader.py              # load_hotpotqa(), get_supporting_sentences()
│   │   ├── distortion.py          # corrupt_fact(), create_noisy_context(), generate_wrong_answer()
│   │   ├── prompting.py           # build_prompt(), build_adversarial_prompt()
│   │   └── verification.py        # generate_with_verification()
│   ├── evaluation/
│   │   └── metrics.py             # is_correct(), is_self_contradictory(), classify_verification_effect()
│   └── experiment.py              # ExperimentRunner -- orchestrates all conditions across the dataset
└── results/
├── experiment_results.csv     # 150 rows: 50 examples x 3 conditions
├── smoke_test.csv             # 3-example pilot run
├── accuracy_by_condition.png
├── contradiction_rate_by_condition.png
└── verification_breakdown_by_condition.png

## Setup
- Python 3.11 (tested on 3.11.5)
- A local [Ollama](https://ollama.com) installation

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.1:8b
```

## Reproducing Results
All experiments run from `notebooks/02_explore_data.ipynb`, top to bottom:

1. **Connection test** (`01_test_connection.ipynb`) -- verifies Ollama is reachable
2. **Data exploration** -- loads one HotpotQA example, inspects its structure
3. **Pipeline construction** -- builds and tests baseline / noisy / adversarial prompts on a single example
4. **Smoke test** -- runs the full pipeline on 3 examples (`results/smoke_test.csv`)
5. **Full experiment** -- runs all 50 examples x 3 conditions (`results/experiment_results.csv`); ~1.5-2 hours on a CPU-only laptop
6. **Scoring & figures** -- computes accuracy and self-contradiction metrics, regenerates all three plots in `results/`

Only step 5 is long-running; steps 1-4 and 6 take seconds to minutes.

## Metrics
**Accuracy** -- a response is correct if its stated conclusion and the
gold answer share a fully-contained set of normalized words in either
direction. Scored against the model's *stated conclusion* (text after
markers like "therefore" / "final answer is"), not the full response,
to avoid crediting a correct fact mentioned in passing while reasoning
toward a different, wrong, final answer.

**Self-contradiction** -- flags responses where scoring the full text
disagrees with scoring only the stated conclusion (i.e. the response
supports one answer while reasoning, then explicitly concludes another).

## Data Sources
HotpotQA (distractor setting) -- Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W. W., Salakhutdinov, R., & Manning, C. D. (2018). [hotpotqa.github.io](https://hotpotqa.github.io/)

## References
- Yang, Z., et al. (2018). HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. *EMNLP*.
- Lin, S., Hilton, J., & Evans, O. (2022). TruthfulQA: Measuring How Models Mimic Human Falsehoods. *ACL*.
- Meng, K., Bau, D., Andonian, A., & Belinkov, Y. (2022). Locating and Editing Factual Associations in GPT. *NeurIPS*.
- Huang, J., Chen, X., Mishra, S., Zheng, H.S., Yu, A.W., Song, X., & Zhou, D. (2024). Large Language Models Cannot Self-Correct Reasoning Yet. *ICLR*. arXiv:2310.01798.

## AI Usage Disclaimer
Parts of this project code structure, debugging, drafting of
descriptive text were developed with the assistance of Claude
(Anthropic). All content was reviewed, tested and validated by me.
I take full responsibility for the final content, its accuracy and
its academic integrity.