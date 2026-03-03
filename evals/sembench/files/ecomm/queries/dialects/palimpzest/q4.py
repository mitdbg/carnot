import os
import pandas as pd
import palimpzest as pz


def run(pz_config, data_dir: str):
    # Load data
    images = pz.ImageFileDataset(
        id="images", path=os.path.join(data_dir, "images")
    )
    styles_details = pd.read_parquet(
        os.path.join(data_dir, "styles_details.parquet")
    )

    # Pre-filter for simple colors
    images = images.add_columns(
        udf=lambda row: {"product_id": row["filename"].split(".", 1)[0]},
        cols=[
            {
                "name": "product_id",
                "type": str,
                "description": "Product id generated from image name",
            }
        ],
    )
    styles_details = styles_details[
        styles_details["baseColour"].isin(
            ["Black", "Blue", "Red", "White", "Orange", "Green"]
        )
    ]
    images = images.filter(
        lambda row: int(row["product_id"]) in styles_details["id"].values
    )

    # Process data
    images = images.sem_add_columns(
        cols=[
            {
                "name": "category",
                "type": str,
                "description": "Extract the primary color of the product in the image. Only return the base color, nothing else.",
            }
        ],
        depends_on=["contents"],
    )
    images = images.project(["product_id", "category"])

    output = images.run(pz_config)
    return output
