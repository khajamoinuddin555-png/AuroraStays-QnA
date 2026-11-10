import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

print("=" * 50)
print("AURORASTAYS QnA - API CONFIGURATION TEST")
print("=" * 50)

if api_key and api_key.startswith("sk-"):
    print("SUCCESS: OpenAI API key loaded.")
else:
    print("ERROR: API key missing or invalid format.")

print("=" * 50)