import os
import pandas as pd
import palimpzest as pz


def run(pz_config, data_dir: str):
    # Load data
    styles_details = pd.read_parquet(
        os.path.join(data_dir, "styles_details.parquet")
    ).rename(
        columns={"id": "prod_id"}
    )  # prevent naming conflict with internal Palimpzest 'id' column
    images1 = pz.ImageFileDataset(
        id="images1", path=os.path.join(data_dir, "images")
    )
    images2 = pz.ImageFileDataset(
        id="images1", path=os.path.join(data_dir, "images")
    )

    # Pre-filter data.
    # Then propagate this filter to 'images' based on the product id.
    images1 = images1.add_columns(
        udf=lambda row: {"prod_id": row["filename"].split(".", 1)[0]},
        cols=[
            {
                "name": "prod_id",
                "type": str,
                "description": "Product id generated from image name",
            }
        ],
    )
    images2 = images2.add_columns(
        udf=lambda row: {"prod_id": row["filename"].split(".", 1)[0]},
        cols=[
            {
                "name": "prod_id",
                "type": str,
                "description": "Product id generated from image name",
            }
        ],
    )
    styles_details = styles_details[
        styles_details["baseColour"].isin(
            ["Black", "Blue", "Red", "White", "Orange", "Green"]
        )
        & (styles_details["colour1"] == "")
        & (styles_details["colour2"] == "")
        & (styles_details["price"] < 800)
    ]
    images1 = images1.filter(
        lambda row: int(row["prod_id"]) in styles_details["prod_id"].values
    )
    images2 = images2.filter(
        lambda row: int(row["prod_id"]) in styles_details["prod_id"].values
    )

    # Join data: image-to-image join; remove self-joiners
    processed = images1.sem_join(
        images2,
        """
        Determine whether both images display objects of the same category
        (e.g., both are shoes, both are bags, etc.) and whether these objects
        share the same dominant surface color. Disregard any logos, text, or
        printed graphics on the objects. There might be other objects in the
        images. Only focus on the main object. Base your comparison solely on
        object type and overall surface color.
        """,
        depends_on=[
            "contents",
            "contents_right",
        ],
    )
    processed = processed.filter(
        lambda row: row["prod_id"] != row["prod_id_right"]
    )

    # Generate joined identifiers
    processed = processed.add_columns(
        udf=lambda row: {
            "product_id": str(row["prod_id"]) + "-" + str(row["prod_id_right"])
        },
        cols=[
            {"name": "product_id", "type": str, "description": "Combined ID"}
        ],
    )
    processed = processed.project(["product_id"])

    output = processed.run(pz_config)
    return output
