import os
import pandas as pd
import palimpzest as pz


def add_image_data(
    styles_details: pz.MemoryDataset, data_dir: str
) -> pz.MemoryDataset:
    styles_details = styles_details.add_columns(
        udf=lambda row: {
            "image_file_path": os.path.join(
                data_dir, "images", str(row["prod_id"]) + ".jpg"
            )
        },
        cols=[
            {
                "name": "image_file_path",
                "type": pz.ImageFilepath,
                "description": "",  # leave empty because this influences the LLMs decision
            }
        ],
    )
    return styles_details


def run(pz_config, data_dir: str):
    # Load data
    styles_details = pd.read_parquet(
        os.path.join(data_dir, "styles_details.parquet")
    ).rename(
        columns={"id": "prod_id"}
    )  # prevent naming conflict with internal Palimpzest 'id' column

    styles_details = styles_details[styles_details.apply(lambda row: row['masterCategory']['typeName'] in ('Accessories', 'Apparel', 'Footwear'), axis=1)]
    
    styles_details = pz.MemoryDataset(id="styles_details", vals=styles_details)
    styles_details = add_image_data(styles_details, data_dir)

    # Filter data
    styles_details = styles_details.sem_filter(
        "Does the following description describe a product from either Adidas or Puma?",
        depends_on=[
            "productDisplayName",
            "productDescriptors",
        ],
    )

    # Project JSON
    styles_details = styles_details.sem_add_columns(
        cols=[
            {
                "name": "product_id",
                "type": str,
                "description": """
                    You are given a product description and an image of the product as well as the product id.
                    The product contains a fashion item (clothing, shoes, accessories, etc).
                    There might be multiple fashion items in the image, especially when a model is presenting them.
                    If this is the case, focus only on the primary fashion item and use the description to determine which item in the image is of interest.

                    For each product, generate the following JSON:
                    ```
                    {
                    "id": <product id> (integer),
                    "brand": <extract the brand name from the description and/or image. use lower-case letters for the brand name>",
                    "category": <classify the images into 'accessories', 'apparel', 'footwear'>
                    }
                    ```

                    Output the json in a single line.
                    Keep the order of the keys in the JSON as given in the description.
                    Do not use spaces between { or keys and values in the JSON, i.e., do no use spaces anywhere in the JSON structure.
                    Use normal quotes in the JSON; do not use single quotes.
                """
            }
        ],
        depends_on=["prod_id", "productDisplayName", "productDescriptors", "image_file_path"],
    )
    
    styles_details = styles_details.project(["product_id"])

    output = styles_details.run(pz_config)
    return output
