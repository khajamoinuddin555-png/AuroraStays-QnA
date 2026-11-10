
import os

from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

from deepeval.models.base_model import DeepEvalBaseLLM


# Load project environment variables
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=True)

API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("OPENAI_BASE_URL")
MODEL_NAME = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


class GreatLearningLLM(DeepEvalBaseLLM):

    def __init__(self):
        if not API_KEY:
            raise ValueError("OPENAI_API_KEY is missing in .env")

        if not BASE_URL:
            raise ValueError("OPENAI_BASE_URL is missing in .env")

        self.client = OpenAI(
            api_key=API_KEY,
            base_url=BASE_URL
        )

        self.model_name = MODEL_NAME

    def load_model(self):
        return self.client

    def generate(self, prompt: str, schema=None) -> str:

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        result = response.choices[0].message.content

        if not result:
            raise ValueError("Great Learning LLM returned an empty response.")

        return result

    async def a_generate(self, prompt: str, schema=None) -> str:
        return self.generate(prompt, schema)

    def get_model_name(self):
        return self.model_name


if __name__ == "__main__":

    print("=" * 60)
    print("AURORASTAYS - DEEPEVAL CUSTOM LLM TEST")
    print("=" * 60)

    evaluation_model = GreatLearningLLM()

    print("\nEvaluation Model:", evaluation_model.get_model_name())

    test_prompt = """
    Answer this question in one sentence:

    What are the quiet hours at AuroraStay Hotels?
    """

    print("\nSending test request to Great Learning API...")

    answer = evaluation_model.generate(test_prompt)

    print("\nGenerated Response:")
    print(answer)

    print("\nGreat Learning custom evaluation LLM test completed!")
