import torch
from trl import SFTTrainer, SFTConfig
from data import preprocess_function
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_id="microsoft/Phi-3-mini-128k-instruct"):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="cuda",
        torch_dtype="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2"
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def main():
    model, tokenizer = load_model_and_tokenizer("microsoft/Phi-3-mini-128k-instruct")

    dataset = load_dataset("qwedsacf/competition_math")
    dataset = dataset.map(lambda x: preprocess_function(x, tokenizer), remove_columns=list(dataset["train"].features))
    train_test_split = dataset["train"].train_test_split(test_size=0.1)
    
    config = SFTConfig(
        output_dir="checkpoints/phi-math",
        packing=True,
        max_seq_length=2048,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        bf16=True,  # mixed-precision training
        logging_steps=10,
        evaluation_strategy="steps", # Evaluate at regular step intervals
        eval_steps=100
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        tokenizer=tokenizer,
        train_dataset=train_test_split["train"],
        eval_dataset=train_test_split["test"],
        dataset_text_field="text"
    )

    trainer.train()
    trainer.save_model("./checkpoints/sft_final.ckpt")
    print("SFT training complete!")

if __name__ == "__main__":
    main()