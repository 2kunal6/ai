from transformers import pipeline
generator = pipeline('text-generation', model='gpt2')

response = generator("AI models are so smart they can replace my", max_length=10)

print(response)