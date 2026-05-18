import json
from io import StringIO

import pandas as pd


JSON_PATH = "data/officeqa/treasury_bulletins_parsed/jsons/treasury_bulletin_1960_05.json"


# treasury_bulletin_1960_05.json
# HAD TO DO CLEANING

with open(JSON_PATH, "r", encoding="utf-8") as json_file:
    parsed_document = json.load(json_file)

elements = parsed_document["document"]["elements"]
table_count = 0
parsed_dataframe_count = 0
failures = []

for element_index, element in enumerate(elements):
    if element.get("type") != "table":
        continue

    table_count += 1
    table_html = element.get("content")
    page_ids = [
        bbox.get("page_id")
        for bbox in element.get("bbox", [])
        if bbox.get("page_id") is not None
    ]
    page_label = ", ".join(str(page_id) for page_id in page_ids) or "unknown"

    if not table_html:
        failures.append(
            {
                "element_index": element_index,
                "element_id": element.get("id"),
                "page": page_label,
                "error": "empty table content",
            }
        )
        continue

    try:
        dataframes = pd.read_html(StringIO(table_html))
    except Exception as exc:
        failures.append(
            {
                "element_index": element_index,
                "element_id": element.get("id"),
                "page": page_label,
                "error": repr(exc),
            }
        )
        continue

    parsed_dataframe_count += len(dataframes)

    for dataframe_index, dataframe in enumerate(dataframes, start=1):
        columns = [str(column) for column in dataframe.columns]
        preview_columns = columns[:8]
        if len(columns) > len(preview_columns):
            preview_columns.append("...")


print()
print(f"table elements found: {table_count}")
print(f"dataframes parsed: {parsed_dataframe_count}")
print(f"failures: {len(failures)}")

for failure in failures:
    print(
        "  failed: "
        f"element_index={failure['element_index']} "
        f"element_id={failure['element_id']} "
        f"page={failure['page']} "
        f"error={failure['error']}"
    )
