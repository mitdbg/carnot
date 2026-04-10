import os
import pandas as pd
import palimpzest as pz


def add_image_data(
    styles_details: pz.MemoryDataset, data_dir: str, suffix: str
) -> pz.MemoryDataset:
    styles_details = styles_details.add_columns(
        udf=lambda row: {
            "image_file_path"
            + suffix: os.path.join(
                data_dir,
                "images",
                str(row["prod_id" + suffix]) + ".jpg",
            )
        },
        cols=[
            {
                "name": "image_file_path" + suffix,
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

    # Pre-filter data
    styles_details = styles_details[
        (styles_details["baseColour"].isin(["Black", "Blue", "Red", "White"]))
        & (styles_details["price"] <= 1000)
    ]

    footwear = styles_details.copy(deep=True)
    footwear.columns = [col + "_footwear" for col in footwear.columns]
    footwear = pz.MemoryDataset(id="footwear", vals=footwear)
    footwear = add_image_data(footwear, data_dir, "_footwear")

    bottomwear = styles_details.copy(deep=True)
    bottomwear.columns = [col + "_bottomwear" for col in bottomwear.columns]
    bottomwear = pz.MemoryDataset(id="bottomwear", vals=bottomwear)
    bottomwear = add_image_data(bottomwear, data_dir, "_bottomwear")

    topwear = styles_details.copy(deep=True)
    topwear.columns = [col + "_topwear" for col in topwear.columns]
    topwear = pz.MemoryDataset(id="topwear", vals=topwear)
    topwear = add_image_data(topwear, data_dir, "_topwear")

    # Filters
    footwear_f = footwear.sem_filter(
        "The image depicts a (pair of) shoe(s), sandal(s), flip-flop(s). If there are multiple products in the picture, always refer to the most promiment one.",
        depends_on=["image_file_path_footwear"],
    )

    bottomwear_f = bottomwear.sem_filter(
        "The image depicts a piece of apparel that can be worn on the lower part of the body, like pants, shorts, skirts, ... If there are multiple products in the picture, always refer to the most promiment one.",
        depends_on=["image_file_path_bottomwear"],
    )

    topwear_f = topwear.sem_filter(
        "The image depicts a piece of apparel that can be worn on the upper part of the body, like t-shirts, shirts, pullovers, hoodies, but still require some sort of clothing on the lower body, which means, e.g., not a dress. If there are multiple products in the picture, always refer to the most promiment one.",
        depends_on=["image_file_path_topwear"],
    )

    # Perform joins
    processed_1 = footwear_f.sem_join(
        bottomwear_f,
        """
        The images depict products with the same primary base color, e.g., both are black, both are white, and both products are from the same brand.
        """,
        depends_on=[
            "productDisplayName_footwear",
            "productDescriptors_footwear",
            "image_file_path_footwear",
            "productDisplayName_bottomwear",
            "productDescriptors_bottomwear",
            "image_file_path_bottomwear",
        ],
    )

    processed_2 = processed_1.sem_join(
        topwear_f,
        """
        The images depict products with the same primary base color, e.g., both are black, both are white, and both products are from the same brand.
        """,
        depends_on=[
            "productDisplayName_bottomwear",
            "productDescriptors_bottomwear",
            "image_file_path_bottomwear",
            "productDisplayName_topwear",
            "productDescriptors_topwear",
            "image_file_path_topwear",
        ],
    )

    # Generate joined identifiers
    processed_3 = processed_2.add_columns(
        udf=lambda row: {
            "product_id": str(row["prod_id_footwear"])
            + "-"
            + str(row["prod_id_bottomwear"])
            + "-"
            + str(row["prod_id_topwear"])
        },
        cols=[
            {"name": "product_id", "type": str, "description": "Combined ID"}
        ],
    )
    processed_4 = processed_3.project(["product_id"])

    output = processed_4.run(pz_config)
    return output
