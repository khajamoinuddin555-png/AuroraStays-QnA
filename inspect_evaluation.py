
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
EVAL_DIR = BASE_DIR / "data" / "evaluation"

# -----------------------------------
# 1. Inspect evaluation.csv
# -----------------------------------

print("=" * 70)
print("1. EVALUATION.CSV INSPECTION")
print("=" * 70)

csv_path = EVAL_DIR / "evaluation.csv"

df_csv = pd.read_csv(csv_path)

print("\nColumn Names:")
print(df_csv.columns.tolist())

print("\nDataset Shape (Rows, Columns):")
print(df_csv.shape)

print("\nAll Evaluation Records:")
print(df_csv.to_string(index=False))


# -----------------------------------
# 2. Inspect test.xlsx
# -----------------------------------

print("\n" + "=" * 70)
print("2. TEST.XLSX INSPECTION")
print("=" * 70)

excel_path = EVAL_DIR / "test.xlsx"

df_excel = pd.read_excel(excel_path)

print("\nColumn Names:")
print(df_excel.columns.tolist())

print("\nDataset Shape (Rows, Columns):")
print(df_excel.shape)

print("\nAll Test Records:")
print(df_excel.to_string(index=False))


# -----------------------------------
# 3. Inspection completed
# -----------------------------------

print("\n" + "=" * 70)
print("DATASET INSPECTION COMPLETED")
print("=" * 70)
