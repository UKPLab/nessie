from pathlib import Path

PATH_ROOT: Path = Path(__file__).resolve().parents[1]


# Example data
PATH_EXAMPLE_DATA: Path = PATH_ROOT / "example_data"
PATH_EXAMPLE_DATA_TEXT: Path = PATH_EXAMPLE_DATA / "easy_text.tsv"
PATH_EXAMPLE_DATA_TOKEN: Path = PATH_EXAMPLE_DATA / "easy_token.conll"
PATH_EXAMPLE_DATA_SPAN: Path = PATH_EXAMPLE_DATA / "easy_span.conll"
