# evaluate.py

import torch

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

from data import extract_boxed, preprocess_function as preprocess_for_eval

def load_model_for_evaluation(model_id, adapter_path):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quantization_config,
        dtype=torch.bfloat16, # use bfloat16 for inference
    )
    
    model = PeftModel.from_pretrained(model, adapter_path)
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


# evaluate the model
def evaluate_model(model, tokenizer, test_dataset):
    model.eval()
    correct_predictions = 0
    
    progress_bar = tqdm(test_dataset, desc="Evaluating")

    for example in progress_bar:
        prompt_formatted = example['prompt']
        expected_answer = example['expected_answer']
        
        inputs = tokenizer(prompt_formatted, return_tensors="pt").to(model.device)
        
        # generate a single, deterministic response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False, # greedy decoding
                pad_token_id=tokenizer.pad_token_id,
                use_cache=False,  # again, disable KV cache for `DynamicCache`` error
            )
        
        # decode the generated text
        response_ids = outputs[0][inputs['input_ids'].shape[1]:]
        response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
        
        # check if the answer is correct
        generated_answer = extract_boxed(response_text)
        if generated_answer is not None and generated_answer == expected_answer:
            correct_predictions += 1
            
        # update progress bar with current accuracy
        current_accuracy = (correct_predictions / (progress_bar.n + 1)) * 100
        progress_bar.set_postfix({'Accuracy': f'{current_accuracy:.2f}%'})

    # final accuracy
    final_accuracy = (correct_predictions / len(test_dataset)) * 100
    return final_accuracy

def main():
    base_model_id = "microsoft/Phi-3-mini-128k-instruct"
    
    sft_model_path = "./checkpoints/phi-math/checkpoint-1896" 
    grpo_model_path = "./checkpoints/grpo_final"

    print("Loading and preparing test dataset...")
    
    eval_tokenizer = AutoTokenizer.from_pretrained(base_model_id)

    full_dataset = load_dataset("qwedsacf/competition_math", split="train")
    
    # train on unseen slice
    test_dataset_raw = full_dataset.select(range(200, 300))

    test_dataset = test_dataset_raw.map(
        lambda x: preprocess_for_eval(x, eval_tokenizer, sft=False),
        remove_columns=list(test_dataset_raw.features)
    )

    sft_model, sft_tokenizer = load_model_for_evaluation(base_model_id, sft_model_path)
    sft_accuracy = evaluate_model(sft_model, sft_tokenizer, test_dataset)
    print(f"\nFinal Accuracy (SFT Model): {sft_accuracy:.2f}%")

    del sft_model
    torch.cuda.empty_cache()

    grpo_model, grpo_tokenizer = load_model_for_evaluation(base_model_id, grpo_model_path)
    grpo_accuracy = evaluate_model(grpo_model, grpo_tokenizer, test_dataset)
    print(f"\nFinal Accuracy (GRPO Model): {grpo_accuracy:.2f}%")

    print("\n--- Benchmark Summary ---")
    print(f"SFT Accuracy:  {sft_accuracy:.2f}%")
    print(f"GRPO Accuracy: {grpo_accuracy:.2f}%")
    
    improvement = grpo_accuracy - sft_accuracy
    print(f"Improvement from GRPO: {improvement:.2f}%")

if __name__ == "__main__":
    main()

