build:
	docker build -t grpo-dev .

run:
	docker run -it --rm --gpus all -e HF_HOME=/app/.cache -v $$(pwd):/app grpo-dev /bin/bash

train-sft:
	python3 src/sft.py