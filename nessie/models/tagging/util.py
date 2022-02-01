import string
from typing import Any, Dict, List


def featurize_sentence(tokens: List[str]) -> List[Dict[str, Any]]:
    return [featurize_token(tokens, token_idx) for token_idx in range(len(tokens))]


def featurize_token(tokens: List[str], token_idx: int) -> Dict[str, Any]:
    def _featurize_single(prefix, idx: int):
        if idx < 0 or idx >= len(tokens):
            return {}

        word = tokens[idx]

        features = {
            "word": word,
            "is_first": idx == 0,
            "is_last": idx == len(tokens) - 1,
            "is_capitalized": word.istitle(),
            "is_all_caps": word.isupper(),
            "is_all_lower": word.islower(),
            "prefix-1": word[0],
            "prefix-2": word[:2],
            "prefix-3": word[:3],
            "suffix-1": word[-1],
            "suffix-2": word[-2:],
            "suffix-3": word[-3:],
            "mention": word.startswith("@") and len(word) > 1,
            "hashtag": word.startswith("#") and len(word) > 1,
            "prev_word": "" if idx == 0 else tokens[idx - 1],
            "next_word": "" if idx == len(tokens) - 1 else tokens[idx + 1],
            "has_hyphen": "-" in word,
            "is_numeric": word.isdigit(),
            "capitals_inside": word[1:].lower() != word[1:],
            "word.ispunctuation": (word in string.punctuation),
        }
        return {f"{prefix}.{k}": v for k, v in features.items()}

    result = {}
    result.update(_featurize_single("0", token_idx))
    result.update(_featurize_single("-1", token_idx - 1))
    result.update(_featurize_single("+1", token_idx + 1))
    return result
