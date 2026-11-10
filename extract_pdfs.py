
from pathlib import Path
from docling.document_converter import DocumentConverter

# Project paths
BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "data" / "processed"

# Create processed folder
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Initialize PDF converter
converter = DocumentConverter()

# Get all PDF files
pdf_files = sorted(PDF_DIR.glob("*.pdf"))

print("=" * 60)
print("AURORASTAYS QnA - PDF TEXT EXTRACTION")
print("=" * 60)

if not pdf_files:
    raise FileNotFoundError("No PDF files found in data/raw")

print(f"\nTotal PDF files found: {len(pdf_files)}\n")

successful = 0
failed = 0

for pdf_file in pdf_files:

    print(f"Processing: {pdf_file.name}")

    try:
        result = converter.convert(str(pdf_file))

        # Export extracted content as Markdown
        text = result.document.export_to_markdown()

        # Save extracted text
        output_file = OUTPUT_DIR / f"{pdf_file.stem}.md"
        output_file.write_text(text, encoding="utf-8")

        print(f"SUCCESS: {output_file.name}")
        print(f"Extracted characters: {len(text)}\n")

        successful += 1

    except Exception as error:
        print(f"FAILED: {pdf_file.name}")
        print(f"Error: {error}\n")

        failed += 1

print("=" * 60)
print("EXTRACTION SUMMARY")
print("=" * 60)
print(f"Successful: {successful}")
print(f"Failed: {failed}")
print(f"Output folder: {OUTPUT_DIR}")

if failed:
    raise RuntimeError(f"{failed} PDF file(s) failed extraction")

print("\nAll PDF documents extracted successfully!")
