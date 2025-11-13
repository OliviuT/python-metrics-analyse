import os
import sys
from openai import OpenAI
from dotenv import load_dotenv

# 1) Get ready: load .env and grab the API key we need to talk to OpenAI.
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    sys.exit("Set OPENAI_API_KEY in your environment or .env file before running this script.")

#user_prompt = "Hello, let me know how can I test streaming on chat completion API?"
user_prompt = input("Enter your prompt: ")
prompt = user_prompt if user_prompt.strip() != "" else "Hello, let me know how can I test streaming on chat completion API?"


# 2) Describe the conversation we want the model to have.
client = OpenAI(api_key=api_key)
completion = client.chat.completions.create(
  model="gpt-4o-mini",
  messages=[
    {"role": "developer", "content": "You are a helpful assistant."},
    {"role": "user", "content": user_prompt}
  ],
  stream=True
)

collected_chunks = []
for chunk in completion:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
        collected_chunks.append(chunk.choices[0].delta.content)
print("\n")

# 3) Print the full response at the end.
#print("\n\n[Full response]:")
#print("".join(collected_chunks))