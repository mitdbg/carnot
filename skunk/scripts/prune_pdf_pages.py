import sys

from pypdf import PdfReader, PdfWriter


INPUT_PDF = "input.pdf"
OUTPUT_PDF = "output.pdf"
PAGES_TO_KEEP = [1, 14, 15, 16, -1]


input_pdf = sys.argv[1] if len(sys.argv) > 1 else INPUT_PDF
output_pdf = sys.argv[2] if len(sys.argv) > 2 else OUTPUT_PDF

reader = PdfReader(input_pdf)
writer = PdfWriter()

page_count = len(reader.pages)
page_indexes = []

for page_number in PAGES_TO_KEEP:
    if page_number == -1:
        page_index = page_count - 1
    else:
        page_index = page_number - 1

    if 0 <= page_index < page_count and page_index not in page_indexes:
        page_indexes.append(page_index)

for page_index in page_indexes:
    writer.add_page(reader.pages[page_index])

with open(output_pdf, "wb") as output_file:
    writer.write(output_file)

kept_pages = ", ".join(str(page_index + 1) for page_index in page_indexes)
print(f"Wrote {output_pdf}")
print(f"Kept pages: {kept_pages}")
