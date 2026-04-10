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

    accessories = styles_details.copy(deep=True)
    accessories = accessories[accessories["price"] <= 500]
    accessories.columns = [col + "_accessories" for col in accessories.columns]
    accessories = pz.MemoryDataset(id="accessories", vals=accessories)
    accessories = add_image_data(accessories, data_dir, "_accessories")

    # Filters
    footwear_f = footwear.sem_filter("""
        You will receive an image and a description of a product.
        Determine whether the product can be worn on the feet, like shoes, sandals, flip-flops, ...
        The predominant color of the depicted product should be black.
        If there are multiple products in the picture, always refer to the most promiment one.
        The description of the product is as follows.""",
        depends_on=[
            "productDisplayName_footwear",
            "productDescriptors_footwear",
        ],
    )

    bottomwear_f = bottomwear.sem_filter("""
        You will receive an image and a description of a product.
        Determine whether the product can be worn on the lower part of the body, like pants, shorts, skirts, ...
        The predominant color of the depicted product should be black.
        Do not consider swimwear.
        If there are multiple products in the picture, always refer to the most promiment one.
        The description of the product is as follows.""",
        depends_on=[
            "productDisplayName_bottomwear",
            "productDescriptors_bottomwear",
        ],
    )

    topwear_f = topwear.sem_filter("""
        You will receive an image and a description of a product.
        Determine whether the product can be worn on the upper part of the body, like t-shirts, shirts, pullovers, hoodies, but still require some sort of clothing on the lower body, which means, e.g., not a dress.
        The predominant color of the depicted product should be black.
        Do not consider swimwear.
        If there are multiple products in the picture, always refer to the most promiment one.
        The description of the product is as follows.""",
        depends_on=[
            "productDisplayName_topwear",
            "productDescriptors_topwear",
        ],
    )

    accessories_f = accessories.sem_filter("""
        You will receive an image and a description of a product.
        Determine whether the product a watch or some jewellery or a bag.
        A bag might be a handbag or a (gym) backpack or some other type of bag.
        If there are multiple products in the picture, always refer to the most promiment one.
        The description of the product is as follows.""",
        depends_on=[
            "productDisplayName_accessories",
            "productDescriptors_accessories",
        ],
    )

    # Perform joins
    processed_1 = footwear_f.sem_join(
        accessories_f,
        """
        You will receive a description and an image of two products.
        Determine whether they are from the same brand.
        """,
        depends_on=[
            "productDisplayName_footwear",
            "productDescriptors_footwear",
            "image_file_path_footwear",
            "productDisplayName_accessories",
            "productDescriptors_accessories",
            "image_file_path_accessories",
        ],
    )

    processed_2 = topwear_f.sem_join(
        bottomwear_f,
        """
        You will receive a description and an image of two products.
        Determine whether they are from the same brand.
        """,
        depends_on=[
            "productDisplayName_topwear",
            "productDescriptors_topwear",
            "image_file_path_topwear",
            "productDisplayName_bottomwear",
            "productDescriptors_bottomwear",
            "image_file_path_bottomwear",
        ],
    )

    processed_3 = processed_1.sem_join(
        processed_2,
        """
        You will receive a description and an image of two products.
        Determine whether they are from the same brand.
        """,
        depends_on=[
            "productDisplayName_footwear",
            "productDescriptors_footwear",
            "image_file_path_footwear",
            "productDisplayName_topwear",
            "productDescriptors_topwear",
            "image_file_path_topwear",
        ],
    )

    # Generate joined identifiers
    processed_4 = processed_3.add_columns(
        udf=lambda row: {
            "product_id": str(row["prod_id_footwear"])
            + "-"
            + str(row["prod_id_bottomwear"])
            + "-"
            + str(row["prod_id_topwear"])
            + "-"
            + str(row["prod_id_accessories"])
        },
        cols=[
            {"name": "product_id", "type": str, "description": "Combined ID"}
        ],
    )
    processed_5 = processed_4.project(["product_id"])

    output = processed_5.run(pz_config)
    return output
