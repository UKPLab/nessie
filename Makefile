force-cuda113:
	python -m pip install --upgrade torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113

force-faiss-gpu:
	python -m pip install faiss-gpu

black:
	black nessie/ tests/ scripts/

isort:
	isort --profile black nessie/ tests/ scripts/

format: black isort

sphinx:
	sphinx-build -b html docs/source/ docs/build/html -j 4
