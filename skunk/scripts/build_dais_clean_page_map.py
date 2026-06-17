import json
import os


INPUT_JSON_DIR = os.environ.get("DAIS_PARSED_JSON_DIR", "/home/ubuntu/dais/parsed_json")
OUTPUT_DIR = os.environ.get("DAIS_CLEAN_PAGE_OUTPUT_DIR", "/home/ubuntu/dais/cleaned_pages_v1")
CLEAN_PAGE_MAP_PATH = os.path.join(OUTPUT_DIR, "clean_page_map.json")


os.makedirs(OUTPUT_DIR, exist_ok=True)

clean_page_map = {}
json_count = 0
page_count = 0
element_count = 0

for filename in sorted(os.listdir(INPUT_JSON_DIR)):
    if not filename.endswith(".json"):
        continue

    json_count += 1
    json_path = os.path.join(INPUT_JSON_DIR, filename)
    with open(json_path, encoding="utf-8") as f:
        doc = json.load(f)

    pages = {}
    for element_index, element in enumerate(doc["document"]["elements"]):
        page_id = element.get("page_id")
        if page_id is None:
            page_ids = []
            for bbox in element.get("bbox") or []:
                if "page_id" in bbox and bbox["page_id"] not in page_ids:
                    page_ids.append(bbox["page_id"])

            if not page_ids:
                raise ValueError(f"{json_path}: element {element_index} has no page_id")
            if len(page_ids) > 1:
                raise ValueError(
                    f"{json_path}: element {element_index} spans pages {page_ids}"
                )

            page_id = page_ids[0]

        if page_id not in pages:
            pages[page_id] = []
        pages[page_id].append((element_index, element))

    stem = filename[:-5]
    for page_id in sorted(pages, key=lambda value: int(value)):
        output_filename = f"{stem}_{page_id}.txt"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        page_elements = pages[page_id]

        contents = []
        element_id_order = []
        for element_index, element in page_elements:
            content = element.get("content")
            contents.append("" if content is None else str(content))
            element_id_order.append(element_index)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(contents).strip())
            f.write("\n")

        doc_id = f"{stem}_{page_id}"
        clean_page_map[doc_id] = [output_path, element_id_order]
        page_count += 1
        element_count += len(page_elements)

with open(CLEAN_PAGE_MAP_PATH, "w", encoding="utf-8") as f:
    json.dump(clean_page_map, f)

print(f"Processed {json_count} JSON documents.")
print(f"Wrote {page_count} page text files to {OUTPUT_DIR}.")
print(f"Indexed {element_count} elements in {CLEAN_PAGE_MAP_PATH}.")
