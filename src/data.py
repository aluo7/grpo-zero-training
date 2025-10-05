# data.py

import re

from datasets import load_dataset
from transformers import AutoTokenizer

def extract_boxed(text):
    match = re.search(r'\\boxed\{(.*?)\}', text, re.DOTALL)
    return match.group(1).strip() if match else None

def preprocess_function(example, tokenizer, sft=True):
    if sft:
        messages = [
            {
                "role": "user", 
                "content": f"Solve the following math problem, providing a step-by-step chain of thought. Enclose the final answer in \\boxed{{}}.\n\nProblem:\n{example['problem']}"
            },
            {
                "role": "assistant", 
                "content": f"Solution:\n{example['solution']}"
            }
        ]
        
        formatted_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=False
        )
        
        return {"text": formatted_text}
    else:
        messages = [
            {
                "role": "user", 
                "content": f"Solve the following math problem, providing a step-by-step chain of thought. Enclose the final answer in \\boxed{{}}.\n\nProblem:\n{example['problem']}"
            }
        ]  # Prompt only, no solution for RL
        
        formatted_prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        expected = extract_boxed(example['solution'])
        
        return {"prompt": formatted_prompt, "expected_answer": expected}
    
def main():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-128k-instruct")

    dataset = load_dataset("qwedsacf/competition_math")
    dataset = dataset.map(lambda x: preprocess_function(x, tokenizer), remove_columns=list(dataset["train"].features))
    print(next(iter(dataset["train"])))

if __name__ == "__main__":
    main()