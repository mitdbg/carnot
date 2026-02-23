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
    ).rename(
        columns={"id": "product_id"}
    )  # prevent naming conflict with internal Palimpzest 'id' column

    # Preprocess data
    styles_details = styles_details[
        styles_details.apply(
            lambda row: row["masterCategory"]["typeName"] == "Apparel"
            and row["subCategory"]["typeName"]
            not in ["Saree", "Apparel Set", "Loungewear and Nightwear"],
            axis=1,
        )
    ]
    styles_details = pz.MemoryDataset(id="styles_details", vals=styles_details)

    # Perform map/extract
    images = images.sem_add_columns(
        cols=[
            {
                "name": "category",
                "type": str,
                "description": """
        You are given an image of a product. Your task is to classify the product
        into one of the following categories: 
        (1) Dress: A dress is a one-piece outer garment that is worn on the torso, hangs down
                    over the legs, and often consist of a bodice attached to a skirt.
        (2) Bottomwear: Bottomwear refers to clothing worn on the lower part of the body,
                    such as trousers, jeans, skirts, shorts, and leggings.
        (3) Socks: Socks are a type of clothing worn on the feet, typically made of soft fabric,
                    designed to provide comfort and warmth.
        (4) Topwear: Topwear refers to clothing worn on the upper part of the body,
                    such as shirts, blouses, t-shirts, and jackets.
        (5) Innerwear: Innerwear refers to clothing worn beneath outer garments,
                    typically close to the skin, such as underwear, bras, and undershirts.
        When classifying the product, only output the category name, nothing more.
        """,
            }
        ],
        depends_on=["contents"],
    )
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
    images = images.project(["product_id", "category"])

    output = images.run(pz_config)
    return output
