VERIFICATION_CHALLENGE = (
    "Are you sure? Double check your assumptions and the evidence "
    "above, then give your final answer."
)


def generate_with_verification(client, prompt):
    """
    Run a prompt through the model, then follow up with a verification
    challenge in the SAME conversation. Returns (first_answer,
    final_answer) so we can measure whether verification changed anything.
    """
    messages = [{"role": "user", "content": prompt}]
    first_answer = client.chat(messages)

    messages.append({"role": "assistant", "content": first_answer})
    messages.append({"role": "user", "content": VERIFICATION_CHALLENGE})
    final_answer = client.chat(messages)

    return first_answer, final_answer
