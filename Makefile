build:
	docker build -t grpo-dev .

run:
	docker run -it --rm --gpus all -e HF_HOME=/app/.cache -v $$(pwd):/app grpo-dev /bin/bash

tensorboard:
	docker run -it --rm -p 6006:6006 -v $$(pwd):/app grpo-dev \
		tensorboard --logdir /app/runs/grpo --host 0.0.0.0 --port 6006

train-sft:
	python3 src/sft.py

train-grpo:
	python3 src/grpo.py
