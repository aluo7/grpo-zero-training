# grpo-zero-training
Implement GRPO Zero training on the MATH dataset.

### Execution Pipeline

First, build and run your environment. This environment is fully containerized using Docker.

```
make build
make run
```

To run supervised fine-tuning, simply run the following. LLM backbone is set to Phi-3-mini by default:

```
make train-sft
```

To train with GRPO policy, use the following make command:

```
make train-grpo
```