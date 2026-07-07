import re
import string


def normalize_answer(text):
    """Lowercase, strip punctuation/articles, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


CONCLUSION_MARKERS = [
    "so, my final answer is", "so my final answer is",
    "final answer is", "final answer:",
    "therefore,", "therefore:",
    "in conclusion,", "in conclusion:",
]


def extract_conclusion(text):
    """Return text after the LAST conclusion marker, or full text if none found."""
    lowered = text.lower()
    best_pos, best_end = -1, -1
    for marker in CONCLUSION_MARKERS:
        pos = lowered.rfind(marker)
        if pos > best_pos:
            best_pos, best_end = pos, pos + len(marker)
    if best_pos == -1:
        return text
    return text[best_end:].strip()


def is_correct(model_answer, gold_answer, use_conclusion_only=True):
    """Correct if gold's words are fully contained in the model's words, or vice versa."""
    text = extract_conclusion(model_answer) if use_conclusion_only else model_answer
    gold_words = set(normalize_answer(gold_answer).split())
    model_words = set(normalize_answer(text).split())
    return gold_words <= model_words or model_words <= gold_words


def is_self_contradictory(model_answer, gold_answer):
    """
    Flags a response where scoring the FULL text disagrees with scoring
    only its stated CONCLUSION -- the response supports one answer while
    reasoning, then explicitly concludes something else. First surfaced
    by the Oliver Reed case: reasoning said "Prussian," conclusion said
    "German."
    """
    whole_response_correct = is_correct(model_answer, gold_answer, use_conclusion_only=False)
    conclusion_correct = is_correct(model_answer, gold_answer, use_conclusion_only=True)
    return whole_response_correct != conclusion_correct


def classify_verification_effect(first_correct, final_correct):
    """Categorize how the verification follow-up changed correctness."""
    if first_correct and final_correct:
        return "stayed_correct"
    if not first_correct and not final_correct:
        return "stayed_wrong"
    if first_correct and not final_correct:
        return "broken_by_verification"
    return "fixed_by_verification"