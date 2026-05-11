# cp data/officeqa/treasury_bulletin_pdfs/treasury_bulletin_1941_01.pdf data/officeqa-tiny/treasury-bulletin_pdfs/
# cp data/officeqa/treasury_bulletin_pdfs/treasury_bulletin_1966_01.pdf data/officeqa-tiny/treasury-bulletin_pdfs/
# cp data/officeqa/treasury_bulletin_pdfs/treasury_bulletin_1942_01.pdf data/officeqa-tiny/treasury-bulletin_pdfs/
# cp data/officeqa/treasury_bulletin_pdfs/treasury_bulletin_2010_03.pdf data/officeqa-tiny/treasury-bulletin_pdfs/
# cp data/officeqa/treasury_bulletin_pdfs/treasury_bulletin_2021_12.pdf data/officeqa-tiny/treasury-bulletin_pdfs/

# cp data/officeqa/treasury_bulletins_parsed/transformed/treasury_bulletin_1941_01.txt data/officeqa-tiny/treasury-bulletins_parsed/transformed/
# cp data/officeqa/treasury_bulletins_parsed/transformed/treasury_bulletin_1966_01.txt data/officeqa-tiny/treasury-bulletins_parsed/transformed/
# cp data/officeqa/treasury_bulletins_parsed/transformed/treasury_bulletin_1942_01.txt data/officeqa-tiny/treasury-bulletins_parsed/transformed/
# cp data/officeqa/treasury_bulletins_parsed/transformed/treasury_bulletin_2010_03.txt data/officeqa-tiny/treasury-bulletins_parsed/transformed/
# cp data/officeqa/treasury_bulletins_parsed/transformed/treasury_bulletin_2021_12.txt data/officeqa-tiny/treasury-bulletins_parsed/transformed/

INPUT_DIR=data/officeqa/treasury_bulletin_pdfs
OUTPUT_DIR=data/officeqa-tiny/treasury-bulletin_pdfs
for file in treasury_bulletin_1941_01.pdf treasury_bulletin_1966_01.pdf treasury_bulletin_1942_01.pdf treasury_bulletin_2010_03.pdf treasury_bulletin_2021_12.pdf; do
    python skunk/prune_pdf_pages.py "$INPUT_DIR/$file" "$OUTPUT_DIR/$file"
done