from datasets import load_dataset


def load_hotpotqa(split="validation", n=None, seed=42):
    """
    Load the HotpotQA distractor-setting dataset. If n is given, returns
    a reproducible random sample of n examples (shuffled with a fixed
    seed) rather than just the first n -- avoids any bias from whatever
    order the dataset happens to be stored in.
    """
    dataset = load_dataset("hotpotqa/hotpot_qa", "distractor", split=split)
    if n is not None:
        dataset = dataset.shuffle(seed=seed).select(range(n))
    return dataset


def get_supporting_sentences(example):
    """Resolve each supporting fact to its actual sentence text, matched by title name."""
    titles = example["context"]["title"]
    sentences_by_paragraph = example["context"]["sentences"]
    resolved = []
    for sf_title, sf_sent_id in zip(example["supporting_facts"]["title"], example["supporting_facts"]["sent_id"]):
        para_index = titles.index(sf_title)
        resolved.append((sf_title, sentences_by_paragraph[para_index][sf_sent_id]))
    return resolved
