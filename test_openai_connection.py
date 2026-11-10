import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)

# Read API configuration
api_key = os.getenv("OPENAI_API_KEY")
base_url = os.getenv("OPENAI_BASE_URL")

if not api_key:
    raise ValueError("API key not found!")

if not base_url:
    raise ValueError("API Base URL not found!")

# Initialize Great Learning API client
client = OpenAI(
    api_key=api_key,
    base_url=base_url
)

print("API Provider: Great Learning")
print("Base URL:", base_url)

try:
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": "Say hello from AuroraStays QnA."
            }
        ],
        temperature=0
    )

    print("\nSUCCESS: API connection established!")
    print("Response:", response.choices[0].message.content)

except Exception as error:
    print("\nAPI connection failed.")
    print("Error:", error)