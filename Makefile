build:
	docker build -t grpo-dev .

run:
	docker run -it --rm --gpus all -e HF_HOME=/app/.cache -v $$(pwd):/app grpo-dev /bin/bash
