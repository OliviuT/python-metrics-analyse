from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain the theory of relativity in simple terms."}
    ]
)  

print(response.choices[0].message.content)

# This code initializes the OpenAI client, sends a chat completion request to the GPT-4o model,
# and prints the response from the model.
# Make sure to set your OPENAI_API_KEY in the .env file before running this code.
# The request includes a system message to set the assistant's behavior and a user message asking for an explanation of the theory of relativity.
# The response from the model is printed to the console.