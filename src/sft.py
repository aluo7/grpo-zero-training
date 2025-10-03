import torch
from trl import SFTTrainer, SFTConfig
from data import preprocess_function
from datasets import load_dataset
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_model_and_tokenizer(model_id="microsoft/Phi-3-mini-128k-instruct"):
    quantization_config = BitsAndBytesConfig(  # 4-bit quantization
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        dtype="auto",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        quantization_config=quantization_config
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def main():
    model, tokenizer = load_model_and_tokenizer("microsoft/Phi-3-mini-128k-instruct")
    model = prepare_model_for_kbit_training(model)

    dataset = load_dataset("qwedsacf/competition_math")
    dataset = dataset.map(lambda x: preprocess_function(x, tokenizer), remove_columns=list(dataset["train"].features))
    train_test_split = dataset["train"].train_test_split(test_size=0.1)

    peft_config = LoraConfig(  # lora
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules="all-linear", # Target all linear layers
    )
    
    config = SFTConfig(
        output_dir="checkpoints/phi-math",
        packing=True,
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        dataset_text_field="text",
        learning_rate=2e-4,
        bf16=True,  # mixed-precision training
        logging_steps=10,
        # tokenizer=tokenizer,
        eval_steps=100,
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=train_test_split["train"],
        eval_dataset=train_test_split["test"],
        peft_config=peft_config
    )

    trainer.train()
    trainer.save_model("./checkpoints/sft_final.ckpt")
    print("SFT training complete!")

if __name__ == "__main__":
    main()