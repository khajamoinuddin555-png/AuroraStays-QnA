
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env", override=True)

api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")

if not api_key or not base_url:
    raise ValueError("API key or Base URL missing in .env")

client = OpenAI(
    api_key=api_key,
    base_url=base_url
)

print("=" * 50)
print("AVAILABLE GREAT LEARNING MODELS")
print("=" * 50)

models = client.models.list()

for model in models.data:
    print(model.id)

print("=" * 50)
print("Model listing completed!")
