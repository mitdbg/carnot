import argparse
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

from skunk.common import LLMClient
from skunk.config import SkunkConfig
from skunk.corpus import render_page_b64
# Table-truncation logic now lives in `corpus` (also reused on-the-fly by the
# page-index full-text filter); imported here under its original name.
from skunk.corpus import _table_display_texts as _create_element_display_texts

MAX_RETRIES = 3
MAX_WORKERS = 64
SERIALIZE_CLEAN_PAGE_MAP_EVERY_N_PAGES = 100
PAGE_CLEANER_WITH_IMG_PROMPT = """Some of the sentences in this parsed document may be ordered incorrectly. You will be presented with the current ordering of the text content, followed by an image of the PDF page which the text is supposed to transcribe. Please output the correct order of the sentences as a comma separated list of their sentence ids (the number before the colon preceding each sentence). Note that page numbers often appear as <sentence_id>: <number> (or <letter-number>).

{page_contents}"""

PAGE_CLEANER_PROMPT = """Some of the sentences in this parsed document may be ordered incorrectly. Please output the correct order of the sentences as a comma separated list of their sentence ids (the number before the colon preceding each sentence):

{page_contents}"""

def clean_page(page_key, page_elements, pdfs_dir, llm: LLMClient) -> tuple[str, list[int]] | tuple[None, None]:
    # if the page has a single element, we can skip the LLM and just return that element's content as the clean page text
    if len(page_elements) == 1:
        return page_elements[0]['content'], [page_elements[0]['id']]

    try_number = 0
    clean_page_text, sentence_id_order = None, []
    try:
        # truncate table content to save tokens, then build the page contents string.
        display_texts = _create_element_display_texts(page_elements)
        page_contents = "\n".join(
            f"{elt['id']}: {display_texts.get(elt['id'], elt['content'])}"
            for elt in page_elements
        )

        # get the b64 encoded image of the page
        year, month, page_id = page_key.split("-")
        rendered = render_page_b64(
            f"{year}-{month}", int(page_id),
            dpi=75, fmt="jpg", jpg_quality=60, pdf_dir=pdfs_dir,
        )
        pdf_b64 = rendered[1] if rendered else None

        # Conversation state: prior assistant/user pairs are appended to
        # user_text on each retry so the model sees the full feedback chain.
        # (LLMClient.call is single-turn, so we flatten the back-and-forth
        # into the user prompt rather than maintaining a chat history.)
        user_text = PAGE_CLEANER_WITH_IMG_PROMPT.format(page_contents=page_contents)
        images = [("image/jpeg", pdf_b64)]

        while clean_page_text is None and try_number < MAX_RETRIES:
            try_number += 1

            sentence_id_order_str = llm.call(
                system="",
                user=user_text,
                images=images,
            ).text
            try:
                sentence_id_order = [int(s.strip()) for s in sentence_id_order_str.split(",")]
            except Exception as e:
                user_text += (
                    f"\n\nYour previous response was:\n{sentence_id_order_str}\n\n"
                    f"Sorry, I couldn't parse the sentence ids from that response. I got the following error: {e}\n\n"
                    f"Please make sure to output a comma separated list of integers corresponding to the sentence ids. "
                    f"The sentence ids for this page are: {[elt['id'] for elt in page_elements]}"
                )
                continue

            # if the sentence id order doesn't include all sentences, or includes any invalid sentence ids, retry (up to a max number of retries)
            if set(sentence_id_order) != set(elt['id'] for elt in page_elements):
                user_text += (
                    f"\n\nYour previous response was:\n{sentence_id_order_str}\n\n"
                    f"The sentence ids you provided do not match the sentence ids on the page. "
                    f"Please try again and make sure to include all sentence ids in your response. "
                    f"The sentence ids for this page are: {[elt['id'] for elt in page_elements]}"
                )
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

        return clean_page_text, sentence_id_order

    except Exception as e:
        print(f"Error cleaning page ({page_key}): {e}")
        return None, None


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

    # shared Vertex client; LLMClient is thread-safe for concurrent .call() use
    # (each call takes the rate-limiter and creates a fresh request).
    llm = LLMClient(SkunkConfig.from_env())

    # create a mapping from (year, month, page_idx) to the cleaned page's file path
    completed, errored = 0, 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_page_key = {
            executor.submit(clean_page, page_key, page_elements, args.pdfs_dir, llm): page_key
            for page_key, page_elements in dirty_page_map.items()
        }

        with tqdm(total=len(dirty_page_map), desc="Cleaning pages", unit="page") as pbar:
            for future in as_completed(future_to_page_key):
                page_key = future_to_page_key[future]
                clean_page_text, element_id_order = future.result()

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
                pbar.set_postfix(completed=completed, errored=errored)

    # serialize the final clean page map to disk
    with open(clean_page_map_path, "w") as f:
        json.dump(clean_page_map, f)

    print(f"Done. Completed {completed}, errored {errored}.")
