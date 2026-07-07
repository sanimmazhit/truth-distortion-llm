import copy


def corrupt_fact(client, true_sentence, subject):
    """
    Rewrite a true supporting-fact sentence into a plausible-but-false
    version, with the subject's name held fixed.
    """
    corruption_prompt = f"""Rewrite the sentence below so it states something FALSE about {subject}, by changing exactly ONE fact (a date, place, nationality, or role). Do NOT change the name "{subject}" itself. Keep the same style and length. Output ONLY the rewritten sentence -- no quotes, no explanation, no preamble.

Sentence: {true_sentence}

Rewritten sentence:"""
    corrupted = client.generate(corruption_prompt).strip()
    return corrupted.strip('"')


def create_noisy_context(client, example, fact_index=0):
    """
    Return a corrupted copy of example's context, with exactly one
    supporting-fact sentence swapped for a false version. Everything
    else stays byte-identical to the original.
    """
    title = example["supporting_facts"]["title"][fact_index]
    sent_id = example["supporting_facts"]["sent_id"][fact_index]
    para_index = example["context"]["title"].index(title)

    true_sentence = example["context"]["sentences"][para_index][sent_id]
    fake_sentence = corrupt_fact(client, true_sentence, subject=title)

    noisy_context = copy.deepcopy(example["context"])
    noisy_context["sentences"][para_index][sent_id] = fake_sentence

    return noisy_context, para_index, true_sentence, fake_sentence


def generate_wrong_answer(client, question, true_answer):
    """
    Produce a plausible-but-wrong answer to a question.

    Yes/no questions get a direct negation, not an LLM-invented answer --
    asked generically, the LLM tends to produce a related-but-orthogonal
    claim (e.g. answering a nationality question with an ethnicity claim)
    rather than a clean opposite, leaving it ambiguous whether the model
    is engaging with the actual assertion or something else.
    """
    normalized = true_answer.strip().lower()
    if normalized == "yes":
        return "no"
    if normalized == "no":
        return "yes"

    prompt = f"""Question: {question}
The correct answer is: {true_answer}

Give a DIFFERENT, plausible-sounding but INCORRECT answer to this same question. Output ONLY the incorrect answer -- no explanation, no quotes.

Incorrect answer:"""
    wrong = client.generate(prompt).strip()
    return wrong.strip('"')

