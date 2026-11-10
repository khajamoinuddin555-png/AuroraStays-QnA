
import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

# Project directories
BASE_DIR = Path(__file__).resolve().parent
INPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "data" / "chunks"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Chunking configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Initialize text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
    length_function=len
)

# Read Markdown documents
markdown_files = sorted(INPUT_DIR.glob("*.md"))

if not markdown_files:
    raise FileNotFoundError("No Markdown documents found in data/processed")

all_chunks = []

print("=" * 60)
print("AURORASTAYS QnA - DOCUMENT CHUNKING")
print("=" * 60)

for file_path in markdown_files:

    print(f"\nProcessing: {file_path.name}")

    content = file_path.read_text(encoding="utf-8")

    # Split document into chunks
    chunks = text_splitter.split_text(content)

    for index, chunk in enumerate(chunks):

        chunk_record = {
            "chunk_id": f"{file_path.stem}_{index + 1}",
            "source": file_path.name,
            "chunk_number": index + 1,
            "text": chunk
        }

        all_chunks.append(chunk_record)

    print(f"Document characters: {len(content)}")
    print(f"Chunks generated: {len(chunks)}")

# Save all chunks as JSON Lines
output_file = OUTPUT_DIR / "chunks.jsonl"

with output_file.open("w", encoding="utf-8") as file:
    for chunk in all_chunks:
        file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

print("\n" + "=" * 60)
print("CHUNKING SUMMARY")
print("=" * 60)
print(f"Documents processed: {len(markdown_files)}")
print(f"Total chunks generated: {len(all_chunks)}")
print(f"Chunk size: {CHUNK_SIZE}")
print(f"Chunk overlap: {CHUNK_OVERLAP}")
print(f"Output file: {output_file}")

print("\nText chunking completed successfully!")
