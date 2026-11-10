
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

RESULT_FILE = (
    BASE_DIR
    / "data"
    / "evaluation_results"
    / "evaluation_results.csv"
)

df = pd.read_csv(RESULT_FILE)

reason_columns = [
    "faithfulness_reason",
    "contextual_recall_reason",
    "contextual_precision_reason",
    "answer_relevancy_reason"
]

print("=" * 80)
print("AURORASTAYS - EVALUATION REASON ANALYSIS")
print("=" * 80)

for index, row in df.iterrows():

    print("\n" + "=" * 80)
    print(f"QUESTION {index + 1}")
    print("=" * 80)

    print("\nQuestion:")
    print(row["question"])

    for column in reason_columns:

        print(f"\n{column.upper().replace('_', ' ')}:")

        reason = row.get(column)

        if pd.isna(reason):
            print("No explanation available.")
        else:
            print(reason)

print("\n" + "=" * 80)
print("EVALUATION REASON INSPECTION COMPLETED")
print("=" * 80)
