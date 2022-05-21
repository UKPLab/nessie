from nessie.types import BoolArray


def percentage_flagged_score(gold_flags: BoolArray, predicted_flags: BoolArray) -> float:
    return sum(predicted_flags) / len(gold_flags)
