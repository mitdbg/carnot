import json
import numbers
import sys
from io import StringIO

import pandas as pd
from markdownify import markdownify


JSON_FILE = "data/officeqa/treasury_bulletins_parsed/jsons/treasury_bulletin_1939_01.json"
TXT_FILE = "data/officeqa/treasury_bulletins_parsed/transformed/treasury_bulletin_1939_01.txt"

SNIPPET_CHARS = 220

if len(sys.argv) == 3:
    JSON_FILE = sys.argv[1]
    TXT_FILE = sys.argv[2]
elif len(sys.argv) != 1:
    raise SystemExit("Usage: python skunk/check_treasury_json_txt_pair.py [json_file txt_file]")


with open(JSON_FILE, encoding="utf-8") as f:
    parsed_json = json.load(f)

with open(TXT_FILE, encoding="utf-8") as f:
    expected_text = f.read().replace("\r\n", "\n")

converted_blocks = []

for element in parsed_json["document"]["elements"]:
    content = element.get("content")
    if not content:
        continue

    if "<table" in content.lower():
        converted_tables = []
        for dataframe in pd.read_html(StringIO(content)):
            if isinstance(dataframe.columns, pd.MultiIndex):
                headers = [
                    " > ".join(str(part) for part in column if str(part) != "nan")
                    for column in dataframe.columns
                ]
            else:
                headers = [str(column) for column in dataframe.columns]

            non_null_values = []
            for row in dataframe.itertuples(index=False, name=None):
                for value in row:
                    if not pd.isna(value):
                        non_null_values.append(value)

            force_float_values = (
                all(isinstance(column, int) for column in dataframe.columns)
                and dataframe.isna().any().any()
                and non_null_values
                and all(isinstance(value, numbers.Number) for value in non_null_values)
            )

            rows = []
            rows.append("| " + " | ".join(headers) + " |")
            rows.append("| " + " | ".join(["---"] * len(headers)) + " |")

            for row in dataframe.itertuples(index=False, name=None):
                cells = []
                for value in row:
                    if force_float_values and not pd.isna(value):
                        cells.append(f"{float(value):.1f}")
                    else:
                        cells.append(str(value))
                rows.append("| " + " | ".join(cells) + " |")

            converted_tables.append("\n".join(rows))

        converted_blocks.append("\n\n".join(converted_tables))
    elif "<" in content and ">" in content:
        converted_blocks.append(markdownify(content).strip())
    else:
        converted_blocks.append(content)

generated_text = "\n\n".join(converted_blocks) + "\n\n"

print(f"JSON: {JSON_FILE}")
print(f"TXT:  {TXT_FILE}")
print(f"Generated chars: {len(generated_text)}")
print(f"Expected chars:  {len(expected_text)}")
print(f"Exact match:     {generated_text == expected_text}")

if generated_text != expected_text:
    first_diff = min(len(generated_text), len(expected_text))
    for index, (generated_char, expected_char) in enumerate(zip(generated_text, expected_text)):
        if generated_char != expected_char:
            first_diff = index
            break

    start = max(0, first_diff - SNIPPET_CHARS)
    end = first_diff + SNIPPET_CHARS

    print(f"First difference index: {first_diff}")
    print("\nGenerated around first difference:")
    print(repr(generated_text[start:end]))
    print("\nExpected around first difference:")
    print(repr(expected_text[start:end]))
