"""
GRPO (Group Relative Policy Optimization) from scratch
Based on DeepSeek-R1 paper approach
"""
import torch
import torch.nn.functional as F
import copy

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
from data import extract_boxed, preprocess_function

def load_model_and_tokenizer(model_id="microsoft/Phi-3-mini-128k-instruct", 
                              checkpoint_dir="checkpoints/phi-math/checkpoint-1896"):
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quantization_config
    )
    
    model = PeftModel.from_pretrained(model, checkpoint_dir)
    model = prepare_model_for_kbit_training(model)
    model.gradient_checkpointing_enable()
    
    # disable cache due to phi-3-mini DynamicCache dependency
    model.config.use_cache = False
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    return model, tokenizer


# binary reward signal (align with dpo -> single winner, rest are losers)
def compute_reward(completion, expected_answer):
    gen_answer = extract_boxed(completion)
    return 1.0 if gen_answer == expected_answer else 0.0


# generate multiple completions per prompt
def generate_completions(model, tokenizer, prompts, num_generations=4, max_new_tokens=512, temperature=1.0):

    model.eval()
    
    # tokenize prompts
    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    all_completions = []
    all_completion_ids = []
    
    with torch.no_grad():
        for _ in range(num_generations):
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=False,  # disable kv cache
            )
            
            prompt_lengths = inputs['input_ids'].shape[1]
            generated_ids = outputs[:, prompt_lengths:]
            
            completions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            all_completions.append(completions)
            all_completion_ids.append(generated_ids)
    
    # reshape: [num_prompts, num_generations]
    completions_reshaped = [[all_completions[j][i] for j in range(num_generations)] 
                            for i in range(len(prompts))]
    ids_reshaped = [[all_completion_ids[j][i] for j in range(num_generations)] 
                    for i in range(len(prompts))]
    
    return completions_reshaped, ids_reshaped


# compute log probabilities of completions given prompts
def compute_log_probs(model, tokenizer, prompts, completion_ids, requires_grad=False):

    if requires_grad:
        model.train()
    else:
        model.eval()

    batch_log_probs = []
    
    for prompt, comp_ids_list in zip(prompts, completion_ids):
        prompt_log_probs = []
        
        for comp_ids in comp_ids_list:
            prompt_inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            prompt_ids = prompt_inputs['input_ids'].to(model.device)
            
            full_ids = torch.cat([prompt_ids, comp_ids.unsqueeze(0).to(model.device)], dim=1)
            
            # use no_grad context manager for the non-gradient case
            with torch.set_grad_enabled(requires_grad):
                outputs = model(full_ids, use_cache=False)
                logits = outputs.logits
            
            prompt_len = prompt_ids.shape[1] # compute log probs for the completion tokens only
            completion_logits = logits[:, prompt_len-1:-1, :] # shift by 1 for next token prediction
            completion_ids_tensor = full_ids[:, prompt_len:]
            
            log_probs = F.log_softmax(completion_logits, dim=-1)
            
            # use gather to get the log-probs of the actual tokens and sum log probs for this completion
            token_log_probs = torch.gather(log_probs, 2, completion_ids_tensor.unsqueeze(-1)).squeeze(-1)
            total_log_prob = token_log_probs.sum(dim=-1)
            
            if requires_grad:
                prompt_log_probs.append(total_log_prob.squeeze())
            else:
                prompt_log_probs.append(total_log_prob.squeeze().item())
        
        batch_log_probs.append(prompt_log_probs)
    
    return batch_log_probs


# compute loss
def grpo_loss(new_log_probs, old_log_probs, rewards, beta=0.1):

    total_loss = 0.0
    num_pairs = 0

    # new_log_probs -> list of tensors
    # old_log_probs -> list of floats
    for i in range(len(rewards)):
        prompt_rewards = rewards[i]
        
        # if there is no winning response in the group, skip this prompt
        if not any(r > 0 for r in prompt_rewards):
            continue

        # index of the single best response and compute log probs
        winner_idx = np.argmax(prompt_rewards)        
        winner_new_lp = new_log_probs[i][winner_idx]
        winner_old_lp = torch.tensor(old_log_probs[i][winner_idx], device=winner_new_lp.device).detach()

        # iterate through rejected responses in the group
        for j in range(len(prompt_rewards)):
            if winner_idx == j:  # skip winner
                continue

            loser_new_lp = new_log_probs[i][j]
            loser_old_lp = torch.tensor(old_log_probs[i][j], device=loser_new_lp.device).detach()

            log_pi_ratio = winner_new_lp - loser_new_lp  # inspired by DPO
            log_ref_ratio = winner_old_lp - loser_old_lp
            
            # increase the difference between the winner's and loser's log probabilities.
            pi_logr = log_pi_ratio - log_ref_ratio
            loss = -F.logsigmoid(beta * pi_logr)
            
            total_loss += loss
            num_pairs += 1
            
    return total_loss / num_pairs if num_pairs > 0 else torch.tensor(0.0, device=new_log_probs[0][0].device)


# main GRPO training loop
def train_grpo(
    model,
    ref_model,
    tokenizer,
    train_dataset,
    val_dataset=None,
    num_epochs=1,
    batch_size=1,
    num_generations=2,
    learning_rate=5e-6,
    max_new_tokens=512,
    temperature=1.0,
    epsilon=0.2,
    output_dir="checkpoints/grpo_final",
    log_dir="runs/grpo"
):
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    writer = SummaryWriter(log_dir)
    
    model.train()
    global_step = 0
    
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        epoch_loss = 0.0
        epoch_reward = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Training Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            prompts = batch['prompt']
            expected_answers = batch['expected_answer']
            
            # generate completions with current policy
            completions, completion_ids = generate_completions(
                model, tokenizer, prompts, 
                num_generations=num_generations,
                max_new_tokens=max_new_tokens,
                temperature=temperature
            )
            
            # compute rewards
            rewards = []
            for prompt_completions, expected in zip(completions, expected_answers):
                prompt_rewards = [compute_reward(comp, expected) for comp in prompt_completions]
                rewards.append(prompt_rewards)
                epoch_reward += sum(prompt_rewards) / len(prompt_rewards)
            
            old_log_probs = compute_log_probs(ref_model, tokenizer, prompts, completion_ids, requires_grad=False)
            new_log_probs = compute_log_probs(model, tokenizer, prompts, completion_ids, requires_grad=True)

            # compute loss
            optimizer.zero_grad()
            loss = grpo_loss(new_log_probs, old_log_probs, rewards)

            # backprop
            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            
            epoch_loss += loss.item()
            global_step += 1
            
            # logging
            avg_reward = sum([sum(r) for r in rewards]) / sum([len(r) for r in rewards])
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_reward': f'{avg_reward:.4f}'})
            
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/avg_reward", avg_reward, global_step)
            
            if global_step % 50 == 0:
                print(f"\nStep {global_step} | Loss: {loss.item():.4f} | Avg Reward: {avg_reward:.4f}")
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        avg_epoch_reward = epoch_reward / len(dataloader)
        print(f"Epoch {epoch + 1} - Avg Loss: {avg_epoch_loss:.4f} | Avg Reward: {avg_epoch_reward:.4f}")
        writer.add_scalar("train/epoch_loss", avg_epoch_loss, epoch)
        writer.add_scalar("train/epoch_reward", avg_epoch_reward, epoch)

        # validation step
        if val_dataset is not None:
            model.eval()
            val_prompts = list(val_dataset['prompt'])
            val_expected = list(val_dataset['expected_answer'])
            val_completions, val_completion_ids = generate_completions(
                model, tokenizer, val_prompts, num_generations=2, max_new_tokens=max_new_tokens
            )
            val_rewards = []
            for comp_list, exp in zip(val_completions, val_expected):
                val_rewards.append([compute_reward(c, exp) for c in comp_list])
            avg_val_reward = sum([sum(r) for r in val_rewards]) / sum([len(r) for r in val_rewards])
            print(f"Validation reward after epoch {epoch+1}: {avg_val_reward:.4f}")
            writer.add_scalar("val/avg_reward", avg_val_reward, epoch)
            model.train()
    
    # save final model
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"\nModel saved to {output_dir}")
    writer.close()


def main():
    print("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer()
    ref_model = copy.deepcopy(model)  # reference model with frozen weights
    
    print("Loading dataset...")
    dataset = load_dataset("qwedsacf/competition_math")
    
    # preprocess for GRPO
    def preprocess_for_grpo(examples):
        processed = []
        for i in range(len(examples['problem'])):
            example = {k: examples[k][i] for k in examples.keys()}
            result = preprocess_function(example, tokenizer, sft=False)
            processed.append(result)
        
        keys = processed[0].keys()
        return {key: [d[key] for d in processed] for key in keys}
    
    dataset = dataset.map(
        preprocess_for_grpo,
        batched=True,
        remove_columns=list(dataset["train"].features)
    )

    # dummy test
    # train_dataset = dataset["train"].select(range(4))
    # val_dataset = dataset["train"].select(range(4, 5))

    # train_grpo(
    #     model=model,
    #     tokenizer=tokenizer,
    #     train_dataset=train_dataset,
    #     val_dataset=val_dataset,
    #     num_epochs=1,
    #     batch_size=2,           # 2 prompts per batch
    #     num_generations=2,      # 2 completions per prompt (reduced from 4)
    #     learning_rate=5e-6,
    #     max_new_tokens=128,     # Shorter generations (reduced from 512)
    #     temperature=1.0,
    #     epsilon=0.2,
    #     output_dir="checkpoints/grpo_final"
    # )

    train_dataset = dataset["train"].select(range(min(100, len(dataset["train"]))))
    val_dataset = dataset["train"].select(range(100, 120))
    
    print(f"Training on {len(train_dataset)} examples, validating on {len(val_dataset)} examples")
    print("Starting GRPO training...")
    

    train_grpo(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=1,
        batch_size=1,
        num_generations=8,
        learning_rate=1e-6,
        max_new_tokens=512,
        temperature=0.4,
        epsilon=0.2,
        output_dir="checkpoints/grpo_final",
        log_dir="runs/grpo"
    )
    
    print("\nGRPO training complete!")


if __name__ == "__main__":
    main()