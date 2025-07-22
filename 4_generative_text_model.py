from transformers import pipeline

# Load GPT-2 based text generation pipeline
generator = pipeline("text-generation", model="gpt2")

# Generate text
prompt = "Life in a world that runs on renewable energy"
output = generator(prompt, max_length=150, num_return_sequences=2)

# Print result
print(output[0]['generated_text'])

