
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
CHUNKS_FILE = BASE_DIR / "data" / "chunks" / "chunks.jsonl"

target_documents = [
    "aurorastay_guest_conduct_accessibility_and_resolution_guidelines.md",
    "aurorastay_refunds_and_additional_charges_guide.md"
]

with open(CHUNKS_FILE, "r", encoding="utf-8") as file:

    chunks = [json.loads(line) for line in file if line.strip()]

for document in target_documents:

    print("\n" + "=" * 80)
    print("DOCUMENT:", document)
    print("=" * 80)

    document_chunks = [
        chunk for chunk in chunks
        if chunk["source"] == document
    ]

    for chunk in document_chunks:

        print("\n" + "-" * 70)
        print("Chunk ID:", chunk["chunk_id"])
        print("Chunk Number:", chunk["chunk_number"])
        print("Content:")
        print(chunk["text"])

print("\n" + "=" * 80)
print("DOCUMENT CHUNK INSPECTION COMPLETED")
print("=" * 80)
