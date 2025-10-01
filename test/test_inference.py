import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset

torch.random.manual_seed(0)

# --- load model and tokenizer ---
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-128k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
    attn_implementation="flash_attention_2"
)
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

# --- load dataset ---
dataset = load_dataset("qwedsacf/competition_math")

example_problem = dataset["train"][5]["problem"]  # test on a single input
prompt = [
    {"role": "user", "content": f"Solve the following math problem, providing a step-by-step chain of thought. Enclose the final answer in \\boxed{{}}.\n\nProblem:\n{example_problem}"},
    {"role": "assistant", "content": "Solution:\n"},
]

prompt_string = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True, return_tensors="pt")
tokenized_prompt = tokenizer(prompt_string, return_tensors="pt").to(model.device)

# --- run inference ---
generation_args = { 
    "max_new_tokens": 512,
    "do_sample": False,  # force greedy decoding - good for math
}

outputs = model.generate(**tokenized_prompt, **generation_args)

# --- decode and output result ---
output_text = tokenizer.decode(outputs[0][tokenized_prompt['input_ids'].shape[1]:], skip_special_tokens=True)

print("--- PROBLEM ---")
print(example_problem)
print("\n--- MODEL OUTPUT ---")
print(output_text)
