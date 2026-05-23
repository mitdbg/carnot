import argparse
import base64
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz
import litellm
from tqdm import tqdm

MAX_RETRIES = 3
MAX_WORKERS = 64
SERIALIZE_CLEAN_PAGE_MAP_EVERY_N_PAGES = 100
TRUNCATE_TABLE_CHARS = 50
PAGE_CLEANER_MODEL = "openai/gpt-5.4"
PAGE_CLEANER_WITH_IMG_PROMPT = """Some of the sentences in this parsed document may be ordered incorrectly. You will be presented with the current ordering of the text content, followed by an image of the PDF page which the text is supposed to transcribe. Please output the correct order of the sentences as a comma separated list of their sentence ids (the number before the colon preceding each sentence). Note that page numbers often appear as <sentence_id>: <number> (or <letter-number>).

{page_contents}"""

PAGE_CLEANER_PROMPT = """Some of the sentences in this parsed document may be ordered incorrectly. Please output the correct order of the sentences as a comma separated list of their sentence ids (the number before the colon preceding each sentence):

{page_contents}"""

def _create_element_display_texts(page_elements):
    """Return a dict mapping table element id -> display string for every element.

    For each table, uses whichever is shorter: its first row or its first TRUNCATE_TABLE_CHARS chars.
    If multiple tables on the page share the same candidate, extends to the shortest
    unique prefix. Appends '...(truncated)' when the content was shortened.
    """
    def _short_cand(content):
        row = content.split('\n')[0]
        return row if len(row) <= TRUNCATE_TABLE_CHARS else content[:TRUNCATE_TABLE_CHARS]

    # initialize display_texts with all non-table elements (which will be fully included in the prompt)
    display_texts = {elt['id']: elt['content'] for elt in page_elements if elt['type'] != 'table'}

    # for table elements, truncate the context and add to display texts
    table_elts = [elt for elt in page_elements if elt['type'] == 'table']
    candidates = {e['id']: _short_cand(e['content']) for e in table_elts}

    if len(table_elts) > 1:
        cand_values = list(candidates.values())
        for elt in table_elts:
            if cand_values.count(candidates[elt['id']]) > 1:
                content = elt['content']
                other_contents = [e['content'] for e in table_elts if e['id'] != elt['id']]
                for length in range(len(candidates[elt['id']]) + 1, len(content) + 1):
                    prefix = content[:length]
                    if not any(other.startswith(prefix) for other in other_contents):
                        candidates[elt['id']] = prefix
                        break

    for elt in table_elts:
        cand = candidates[elt['id']]
        display_texts[elt['id']] = cand + '...(truncated)' if len(cand) < len(elt['content']) else elt['content']

    return display_texts

def _render_pdf_page_b64(year, month, page_id, pdfs_dir, dpi=75):
    """Render a single PDF page (1-indexed page_id) and return a base64 JPEG string."""
    pdf_path = os.path.join(pdfs_dir, f"treasury_bulletin_{year}_{month}.pdf")
    doc = fitz.open(pdf_path)
    page = doc[int(page_id) - 1]  # fitz is 0-indexed; page_id may be a string
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat)
    return base64.b64encode(pix.tobytes("jpg", jpg_quality=60)).decode()

def clean_page(page_key, page_elements, pdfs_dir) -> tuple[str, list[int], float] | tuple[None, None, float]:
    # if the page has a single element, we can skip the LLM and just return that element's content as the clean page text
    if len(page_elements) == 1:
        return page_elements[0]['content'], [page_elements[0]['id']], 0.0

    try_number = 0
    clean_page_text, sentence_id_order, total_cost = None, [], 0.0
    try:
        # truncate table content to save tokens, then build the page contents string.
        display_texts = _create_element_display_texts(page_elements)
        page_contents = "\n".join(
            f"{elt['id']}: {display_texts.get(elt['id'], elt['content'])}"
            for elt in page_elements
        )

        # get the b64 encoded image of the page
        year, month, page_id = page_key.split("-")
        pdf_b64 = _render_pdf_page_b64(year, month, page_id, pdfs_dir)

        messages = [
            {"role": "user", "content": [
                {"type": "text", "text": PAGE_CLEANER_WITH_IMG_PROMPT.format(page_contents=page_contents)},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{pdf_b64}"}},
            ]}
        ]

        while clean_page_text is None and try_number < MAX_RETRIES:
            try_number += 1

            # call the LLM to get the correct order of sentences
            response = litellm.completion(
                messages=messages,
                model=PAGE_CLEANER_MODEL,
            )
            total_cost += response._hidden_params['response_cost']
            sentence_id_order_str: str = response.choices[0].message.content # type: ignore
            try:
                sentence_id_order = [int(s.strip()) for s in sentence_id_order_str.split(",")]
            except Exception as e:
                messages.append({"role": "assistant", "content": sentence_id_order_str})
                messages.append({"role": "user", "content": f"Sorry, I couldn't parse the sentence ids from your response. I got the following error: {e}\n\nPlease make sure to output a comma separated list of integers corresponding to the sentence ids. The sentence ids for this page are: {[elt['id'] for elt in page_elements]}"})
                continue

            # if the sentence id order doesn't include all sentences, or includes any invalid sentence ids, retry (up to a max number of retries)
            if set(sentence_id_order) != set(elt['id'] for elt in page_elements):
                messages.append({"role": "assistant", "content": sentence_id_order_str})
                messages.append({"role": "user", "content": f"The sentence ids you provided do not match the sentence ids on the page. Please try again and make sure to include all sentence ids in your response. The sentence ids for this page are: {[elt['id'] for elt in page_elements]}"})
                continue

            # construct the clean page text by concatenating the sentences in the correct order
            clean_page_text = ""
            for sentence_id in sentence_id_order:
                for elt in page_elements:
                    if elt['id'] == sentence_id:
                        clean_page_text += elt['content'] + "\n"
                        break

        if clean_page_text is None:
            raise Exception(f"Failed to clean page after {MAX_RETRIES} retries.")

        return clean_page_text, sentence_id_order, total_cost

    except Exception as e:
        print(f"Error cleaning page ({page_key}): {e}")
        return None, None, total_cost


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Produce a cleaner version of each page txt.")
    parser.add_argument("--input-json-dir", help="Path to the directory with the input document JSONs.", default="treasury_bulletins_parsed/jsons")
    parser.add_argument("--pdfs-dir", help="Path to the directory with the input PDFs.", default="treasury_bulletin_pdfs")
    parser.add_argument("--output-dir", help="Path to the output directory where the cleaned page txt files will be saved.", default="treasury_bulletins_cleaned")
    args = parser.parse_args()

    # check that the input directory exists
    if not os.path.isdir(args.input_json_dir):
        print(f"Error: Input directory {args.input_json_dir} does not exist.")
        exit(1)

    # create the output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # load the clean page map if it already exists (in case we need to resume from a previous run)
    clean_page_map = {}
    clean_page_map_path = os.path.join(args.output_dir, "clean_page_map.json")
    if os.path.isfile(clean_page_map_path):
        with open(clean_page_map_path) as f:
            clean_page_map = json.load(f)

    # walk the input directory and get the dirty page data from each JSON document
    dirty_page_map = {}
    for _, _, files in sorted(os.walk(args.input_json_dir)):
        for file in files:
            if file.endswith(".json"):
                # read the JSON document
                json_path = os.path.join(args.input_json_dir, file)
                with open(json_path) as f:
                    doc = json.load(f)

                # construct mapping from page_id --> list of elements on that page
                page_id_to_elements = {}
                for elt in doc['document']['elements']:
                    # NOTE: I checked and all elements should be on a single page; keeping this check just in case
                    if len(elt['bbox']) > 1 and len(set(d['page_id'] for d in elt['bbox'])) > 1:
                        raise Exception(f"File {file} Element {elt['id']} has bounding boxes on multiple pages.")

                    # NOTE: I checked, and only figures have None content values; for all other element types
                    if elt['content'] is None and elt['type'] == "figure":
                        elt['content'] = f"<figure id={elt['id']}>"

                    page_id = elt['bbox'][0]['page_id']
                    if page_id not in page_id_to_elements:
                        page_id_to_elements[page_id] = []
                    page_id_to_elements[page_id].append(elt)

                # use regex to extract year and month from filename with format treasury_bulletin_yyyy_mm.json
                match = re.match(r"treasury_bulletin_(\d{4})_(\d{2})\.json", file)
                year, month = match.groups() # type: ignore
                
                # store f"{year}_{month}_{page_id}" (doc_id format) --> list of elements on that page
                for page_id, elements in page_id_to_elements.items():
                    page_key = f"{year}_{month}_{page_id}"
                    if page_key not in clean_page_map:
                        dirty_page_map[page_key] = elements

    # create a mapping from (year, month, page_idx) to the cleaned page's file path
    completed, errored, total_cost = 0, 0, 0.0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_page_key = {
            executor.submit(clean_page, page_key, page_elements, args.pdfs_dir): page_key
            for page_key, page_elements in dirty_page_map.items()
        }

        with tqdm(total=len(dirty_page_map), desc="Cleaning pages", unit="page") as pbar:
            for future in as_completed(future_to_page_key):
                page_key = future_to_page_key[future]
                clean_page_text, element_id_order, dollar_cost = future.result()

                # always add cost; to capture parsing failures
                total_cost += dollar_cost

                # write clean page to disk and then add to the clean page map
                if clean_page_text is not None:
                    completed += 1
                    year, month, page_id = page_key.split("-")
                    clean_page_path = os.path.join(args.output_dir, f"treasury_bulletin_{year}_{month}_{page_id}.txt")
                    with open(clean_page_path, "w") as f:
                        f.write(clean_page_text)
                    clean_page_map[page_key] = [clean_page_path, element_id_order]

                    # serialize the clean page map to disk every N pages
                    if len(clean_page_map) % SERIALIZE_CLEAN_PAGE_MAP_EVERY_N_PAGES == 0:
                        with open(clean_page_map_path, "w") as f:
                            json.dump(clean_page_map, f)
                else:
                    errored += 1

                pbar.update(1)
                pbar.set_postfix(completed=completed, errored=errored, cost=f"${total_cost:.4f}")

                # if completed == 100:
                #     executor.shutdown(wait=False, cancel_futures=True)
                #     break

    # serialize the final clean page map to disk
    with open(clean_page_map_path, "w") as f:
        json.dump(clean_page_map, f)

    print(f"Done. Total cost: ${total_cost:.4f}")
