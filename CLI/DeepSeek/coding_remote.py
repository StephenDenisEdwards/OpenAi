from huggingface_hub import InferenceClient

# Replace with your actual Hugging Face token
hf_token = "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"

# Choose the model you want to use
# model_id = "deepseek-ai/deepseek-coder-6.7b-base"
model_id = "deepseek-ai/deepseek-coder-1.3b-base"
# Initialize the client with your token
client = InferenceClient(model=model_id, token=hf_token)

# Define your prompt
prompt = "Write a Python class for a binary tree."

# Generate text from the model
response = client.text_generation(
    prompt=prompt,
    max_new_tokens=250,
    temperature=0.7,
    top_p=0.9
)

print(response)

