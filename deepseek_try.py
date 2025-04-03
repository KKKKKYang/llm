from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load Janus model and tokenizer
model_name = "deepseek-ai/Janus-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.eval()

# Prepare prompt context
system_prompt = {"role": "system", "content": "You are an expert in Job Shop Scheduling Problem."}

user_prompt = {
    "role": "user",
    "content": "Optimize schedule for 3 Jobs across 3 Machines to minimize makespan. Each job involves a series of Operations needing specific machines and times. Operations are processed in order, without interruption, on a single Machine at a time.\n\nProblem:\nJob 0 consists of the following Operations:\nOperation 0 on Machine 0 duration 105 mins.\nOperation 1 on Machine 1 duration 29 mins.\nOperation 2 on Machine 2 duration 213 mins.\n\nJob 1 consists of the following Operations:\nOperation 0 on Machine 2 duration 193 mins.\nOperation 1 on Machine 1 duration 18 mins.\nOperation 2 on Machine 0 duration 213 mins.\n\nJob 2 consists of the following Operations:\nOperation 0 on Machine 0 duration 78 mins.\nOperation 1 on Machine 2 duration 74 mins.\nOperation 2 on Machine 1 duration 221 mins."
}

# Convert to Janus format
messages = [system_prompt, user_prompt]
chat_template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Tokenize input
inputs = tokenizer(chat_template, return_tensors="pt").to(model.device)

# Generate solution
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.95,
        do_sample=True
    )

# Decode and print
response = tokenizer.decode(output[0], skip_special_tokens=True)
print("\n=== Model Output ===\n")
print(response)

# Note: For best output, run this in an environment with sufficient memory (>=16GB GPU).
