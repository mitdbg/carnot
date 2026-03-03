"""
Carnot system runner implementation.
Placeholder required by the current structure of the benchmarking framework.
"""

import os
import sys
from pathlib import Path

import carnot

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from runner.generic_carnot_runner.generic_carnot_runner import (
    GenericCarnotRunner,
)


class CarnotRunner(GenericCarnotRunner):
    def __init__(
        self,
        use_case: str,
        scale_factor: int,
        model_name: str = "gemini-2.5-flash",
        concurrent_llm_worker=20,
        skip_setup: bool = False,
    ):
        self.llm_config = {"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")}
        super().__init__(
            use_case,
            scale_factor,
            model_name,
            concurrent_llm_worker,
            skip_setup,
        )
    
    def _execute_q1(self) -> tuple[list[dict], dict]:
        """Q1: Based on the textual description and the title of the product, find the product ids of products that are backpacks from Reebok."""
        styles_df = self.load_data("styles_details.parquet")
        dataset = carnot.Dataset(
            name="Styles Data",
            annotation="",
            items=styles_df.to_dict(orient="records"),
        )
        execution = carnot.Execution(
            query="Based on the textual description and the title of the product, find the product ids of products that are backpacks from Reebok. Return the id of each product.",
            datasets=[dataset],
            llm_config=self.llm_config,
        )
        _, plan = execution.plan()
        execution._plan = plan
        output, _ = execution.run()

        return output, {"total_tokens": 0, "total_execution_cost": 0.0}

    def _execute_q2(self) -> tuple[list[dict], dict]:
        """Q2: Based on the image representation of the product, find the product ids of products where the image shows a pair of sports shoes that are predominantly yellow and silver."""
        # TODO: come back once carnot supports image data
        return [], {"total_tokens": 0, "total_execution_cost": 0.0}
        # images_dir = Path(self.scenario_handler.get_data_dir()) / "images"
        # image_items = [
        #     {"image_path": str(p), "filename": p.name}
        #     for p in images_dir.iterdir()
        #     if p.is_file()
        # ]
        # dataset = carnot.Dataset(
        #     name="Product Images",
        #     annotation=(
        #     "image_path points to the image file; filename is the original file name"
        #     ),
        #     items=image_items,
        # )
        # execution = carnot.Execution(
        #     query=(
        #     "Based on the image, find products where the image shows a pair of sports "
        #     "shoes that are predominantly yellow and silver. Return product_id, "
        #     "defined as filename without the extension. Return the id of each product."
        #     ),
        #     datasets=[dataset],
        #     llm_config=self.llm_config,
        # )
        # _, plan = execution.plan()
        # execution._plan = plan
        # output, _ = execution.run()

        # return output, {"total_tokens": 0, "total_execution_cost": 0.0}

    def _execute_q3(self) -> tuple[list[dict], dict]:
        """Q3: For each product, extract the brand name from the product description and title. Return the product id and the brand in a column titled 'category'."""
        styles_df = self.load_data("styles_details.parquet")
        dataset = carnot.Dataset(
            name="Styles Data",
            annotation="",
            items=styles_df.to_dict(orient="records"),
        )
        execution = carnot.Execution(
            query="For each product, extract the brand name from the product description and title. Return the product id and the brand in a column titled 'category'.",
            datasets=[dataset],
            llm_config=self.llm_config,
        )
        _, plan = execution.plan()
        execution._plan = plan
        output, _ = execution.run()

        return output, {"total_tokens": 0, "total_execution_cost": 0.0}

    def _execute_q4(self) -> tuple[list[dict], dict]:
        """Q4: For each product, extract the primary color of the depicted product. Return the product id and the primary color in a column titled 'category'."""
        # TODO: come back once carnot supports image data
        return [], {"total_tokens": 0, "total_execution_cost": 0.0}
        # # Load data
        # images = pz.ImageFileDataset(
        #     id="images", path=os.path.join(data_dir, "images")
        # )
        # styles_details = pd.read_parquet(
        #     os.path.join(data_dir, "styles_details.parquet")
        # )

        # # Pre-filter for simple colors
        # images = images.add_columns(
        #     udf=lambda row: {"product_id": row["filename"].split(".", 1)[0]},
        #     cols=[
        #         {
        #             "name": "product_id",
        #             "type": str,
        #             "description": "Product id generated from image name",
        #         }
        #     ],
        # )
        # styles_details = styles_details[
        #     styles_details["baseColour"].isin(
        #         ["Black", "Blue", "Red", "White", "Orange", "Green"]
        #     )
        # ]
        # images = images.filter(
        #     lambda row: int(row["product_id"]) in styles_details["id"].values
        # )

        # # Process data
        # images = images.sem_add_columns(
        #     cols=[
        #         {
        #             "name": "category",
        #             "type": str,
        #             "description": "Extract the primary color of the product in the image. Only return the base color, nothing else.",
        #         }
        #     ],
        #     depends_on=["contents"],
        # )
        # images = images.project(["product_id", "category"])

        # output = images.run(pz_config)
        # return output

    def _execute_q5(self) -> tuple[list[dict], dict]:
        """Q5: Based solely on the title and description of the product, classify each product into one of the following categories:

        Dress: A dress is a one-piece outer garment that is worn on the torso, hangs down over the legs, and often consist of a bodice attached to a skirt.
        Bottomwear: Bottomwear refers to clothing worn on the lower part of the body, such as trousers, jeans, skirts, shorts, and leggings.
        Socks: Socks are a type of clothing worn on the feet, typically made of soft fabric, designed to provide comfort and warmth.
        Topwear: Topwear refers to clothing worn on the upper part of the body, such as shirts, blouses, t-shirts, and jackets.
        Innerwear: Innerwear refers to clothing worn beneath outer garments, typically close to the skin, such as underwear, bras, and undershirts.

        Each product can only have one category.
        Return the id of the product as well as its category.
        """
        styles_df = self.load_data("styles_details.parquet")
        styles_df = styles_df[
            styles_df.apply(
                lambda row: row["masterCategory"]["typeName"] == "Apparel"
                and row["subCategory"]["typeName"]
                not in ["Saree", "Apparel Set", "Loungewear and Nightwear"],
                axis=1,
            )
        ]
        dataset = carnot.Dataset(
            name="Styles Data",
            annotation="",
            items=styles_df.to_dict(orient="records"),
        )
        execution = carnot.Execution(
            query=(
                "Based solely on the title and description of the product, classify each product into one of the following categories:\n\n"
                "Dress: A dress is a one-piece outer garment that is worn on the torso, hangs down over the legs, and often consist of a bodice attached to a skirt.\n"
                "Bottomwear: Bottomwear refers to clothing worn on the lower part of the body, such as trousers, jeans, skirts, shorts, and leggings.\n"
                "Socks: Socks are a type of clothing worn on the feet, typically made of soft fabric, designed to provide comfort and warmth.\n"
                "Topwear: Topwear refers to clothing worn on the upper part of the body, such as shirts, blouses, t-shirts, and jackets.\n"
                "Innerwear: Innerwear refers to clothing worn beneath outer garments, typically close to the skin, such as underwear, bras, and undershirts.\n\n"
                "Each product can only have one category. Return the id of the product as well as its category."
            ),
            datasets=[dataset],
            llm_config=self.llm_config,
        )
        _, plan = execution.plan()
        execution._plan = plan
        output, _ = execution.run()

        return output, {"total_tokens": 0, "total_execution_cost": 0.0}
    
    def _execute_q6(self) -> tuple[list[dict], dict]:
        """Q6: Based solely on the image of the product, classify each product into one of the following categories:

        Dress: A dress is a one-piece outer garment that is worn on the torso, hangs down over the legs, and often consist of a bodice attached to a skirt.
        Bottomwear: Bottomwear refers to clothing worn on the lower part of the body, such as trousers, jeans, skirts, shorts, and leggings.
        Socks: Socks are a type of clothing worn on the feet, typically made of soft fabric, designed to provide comfort and warmth.
        Topwear: Topwear refers to clothing worn on the upper part of the body, such as shirts, blouses, t-shirts, and jackets.
        Innerwear: Innerwear refers to clothing worn beneath outer garments, typically close to the skin, such as underwear, bras, and undershirts.

        Each product can only have one category.
        Return the id of the product as well as its category.
        """
        # TODO: come back once carnot supports image data
        return [], {"total_tokens": 0, "total_execution_cost": 0.0}
        # # Load data
        # images = pz.ImageFileDataset(
        #     id="images", path=os.path.join(data_dir, "images")
        # )
        # styles_details = pd.read_parquet(
        #     os.path.join(data_dir, "styles_details.parquet")
        # ).rename(
        #     columns={"id": "product_id"}
        # )  # prevent naming conflict with internal Palimpzest 'id' column

        # # Preprocess data
        # styles_details = styles_details[
        #     styles_details.apply(
        #         lambda row: row["masterCategory"]["typeName"] == "Apparel"
        #         and row["subCategory"]["typeName"]
        #         not in ["Saree", "Apparel Set", "Loungewear and Nightwear"],
        #         axis=1,
        #     )
        # ]
        # styles_details = pz.MemoryDataset(id="styles_details", vals=styles_details)

        # # Perform map/extract
        # images = images.sem_add_columns(
        #     cols=[
        #         {
        #             "name": "category",
        #             "type": str,
        #             "description": """
        #     You are given an image of a product. Your task is to classify the product
        #     into one of the following categories: 
        #     (1) Dress: A dress is a one-piece outer garment that is worn on the torso, hangs down
        #                 over the legs, and often consist of a bodice attached to a skirt.
        #     (2) Bottomwear: Bottomwear refers to clothing worn on the lower part of the body,
        #                 such as trousers, jeans, skirts, shorts, and leggings.
        #     (3) Socks: Socks are a type of clothing worn on the feet, typically made of soft fabric,
        #                 designed to provide comfort and warmth.
        #     (4) Topwear: Topwear refers to clothing worn on the upper part of the body,
        #                 such as shirts, blouses, t-shirts, and jackets.
        #     (5) Innerwear: Innerwear refers to clothing worn beneath outer garments,
        #                 typically close to the skin, such as underwear, bras, and undershirts.
        #     When classifying the product, only output the category name, nothing more.
        #     """,
        #         }
        #     ],
        #     depends_on=["contents"],
        # )
        # images = images.add_columns(
        #     udf=lambda row: {"product_id": row["filename"].split(".", 1)[0]},
        #     cols=[
        #         {
        #             "name": "product_id",
        #             "type": str,
        #             "description": "Product id generated from image name",
        #         }
        #     ],
        # )
        # images = images.project(["product_id", "category"])

        # output = images.run(pz_config)
        # return output

    def _execute_q7(self) -> tuple[list[dict], dict]:
        """Q7: Find all pairs of products priced at $5 or less where both products are of the same category and from the same brand based on their descriptions."""
        styles_df = self.load_data("styles_details.parquet")
        styles_df_right = styles_df.add_suffix("_right")
        dataset1 = carnot.Dataset(
            name="Styles Data 1",
            annotation="",
            items=styles_df.to_dict(orient="records"),
        )
        dataset2 = carnot.Dataset(
            name="Styles Data 2",
            annotation="",
            items=styles_df_right.to_dict(orient="records"),
        )
        execution = carnot.Execution(
            query=(
                "Find all pairs of products priced at $5 or less where both products are of the same category and from the same brand based on their descriptions. Return the product ids of both products in each pair, as well as the shared category and brand."
            ),
            datasets=[dataset1, dataset2],
            llm_config=self.llm_config,
        )
        _, plan = execution.plan()
        execution._plan = plan
        output, _ = execution.run()

        return output, {"total_tokens": 0, "total_execution_cost": 0.0}

    def _execute_q8(self) -> tuple[list[dict], dict]:
        """Q8: Perform a self-join of the dataset. For each product with a product description having at least 3000 characters, find the matching product images based on the title and the description of the product."""
        # TODO: come back once carnot supports image data
        return [], {"total_tokens": 0, "total_execution_cost": 0.0}
        # # Load data
        # styles_details = pd.read_parquet(
        #     os.path.join(data_dir, "styles_details.parquet")
        # ).rename(
        #     columns={"id": "prod_id"}
        # )  # prevent naming conflict with internal Palimpzest 'id' column
        # images = pz.ImageFileDataset(
        #     id="images", path=os.path.join(data_dir, "images")
        # )

        # # Pre-filter data: Filter for long descriptions.
        # # Then propagate this filter to 'images' based on the product id.
        # images = images.add_columns(
        #     udf=lambda row: {"prod_id": row["filename"].split(".", 1)[0]},
        #     cols=[
        #         {
        #             "name": "prod_id",
        #             "type": str,
        #             "description": "Product id generated from image name",
        #         }
        #     ],
        # )
        # styles_details = styles_details[
        #     styles_details.apply(
        #         lambda row: (
        #             row["productDescriptors"].get("description") is not None and
        #             row["productDescriptors"]["description"].get("value") is not None and
        #             len(row["productDescriptors"]["description"]["value"]) >= 3000
        #         ),
        #         axis=1,
        #     )
        # ]
        # # images = images.filter(
        # #     lambda row: int(row["prod_id"]) in styles_details["prod_id"].values
        # # )

        # styles_details_ds = pz.MemoryDataset(
        #     id="styles_details", vals=styles_details
        # )

        # # Join data: text-to-image join
        # processed = styles_details_ds.sem_join(
        #     images,
        #     """
        #     The image fits the description
        #     """,
        #     depends_on=[
        #         "contents",
        #         "productDisplayName",
        #         "productDescriptors",
        #     ],
        # )

        # # Generate joined identifiers
        # processed = processed.add_columns(
        #     udf=lambda row: {
        #         "product_id": str(row["prod_id"]) + "-" + str(row["prod_id_right"])
        #     },
        #     cols=[
        #         {"name": "product_id", "type": str, "description": "Combined ID"}
        #     ],
        # )
        # processed = processed.project(["product_id"])

        # output = processed.run(pz_config)
        # return output

    def _execute_q9(self) -> tuple[list[dict], dict]:
        """Q9: Based on product images, find pairs of distinct products under $8 in a single base color (Black, Blue, Red, White, Orange, or Green) that depict objects of the same category and the same dominant surface color."""
        # TODO: come back once carnot supports image data
        return [], {"total_tokens": 0, "total_execution_cost": 0.0}
        # # Load data
        # styles_details = pd.read_parquet(
        #     os.path.join(data_dir, "styles_details.parquet")
        # ).rename(
        #     columns={"id": "prod_id"}
        # )  # prevent naming conflict with internal Palimpzest 'id' column
        # images1 = pz.ImageFileDataset(
        #     id="images1", path=os.path.join(data_dir, "images")
        # )
        # images2 = pz.ImageFileDataset(
        #     id="images1", path=os.path.join(data_dir, "images")
        # )

        # # Pre-filter data.
        # # Then propagate this filter to 'images' based on the product id.
        # images1 = images1.add_columns(
        #     udf=lambda row: {"prod_id": row["filename"].split(".", 1)[0]},
        #     cols=[
        #         {
        #             "name": "prod_id",
        #             "type": str,
        #             "description": "Product id generated from image name",
        #         }
        #     ],
        # )
        # images2 = images2.add_columns(
        #     udf=lambda row: {"prod_id": row["filename"].split(".", 1)[0]},
        #     cols=[
        #         {
        #             "name": "prod_id",
        #             "type": str,
        #             "description": "Product id generated from image name",
        #         }
        #     ],
        # )
        # styles_details = styles_details[
        #     styles_details["baseColour"].isin(
        #         ["Black", "Blue", "Red", "White", "Orange", "Green"]
        #     )
        #     & (styles_details["colour1"] == "")
        #     & (styles_details["colour2"] == "")
        #     & (styles_details["price"] < 800)
        # ]
        # images1 = images1.filter(
        #     lambda row: int(row["prod_id"]) in styles_details["prod_id"].values
        # )
        # images2 = images2.filter(
        #     lambda row: int(row["prod_id"]) in styles_details["prod_id"].values
        # )

        # # Join data: image-to-image join; remove self-joiners
        # processed = images1.sem_join(
        #     images2,
        #     """
        #     Determine whether both images display objects of the same category
        #     (e.g., both are shoes, both are bags, etc.) and whether these objects
        #     share the same dominant surface color. Disregard any logos, text, or
        #     printed graphics on the objects. There might be other objects in the
        #     images. Only focus on the main object. Base your comparison solely on
        #     object type and overall surface color.
        #     """,
        #     depends_on=[
        #         "contents",
        #         "contents_right",
        #     ],
        # )
        # processed = processed.filter(
        #     lambda row: row["prod_id"] != row["prod_id_right"]
        # )

        # # Generate joined identifiers
        # processed = processed.add_columns(
        #     udf=lambda row: {
        #         "product_id": str(row["prod_id"]) + "-" + str(row["prod_id_right"])
        #     },
        #     cols=[
        #         {"name": "product_id", "type": str, "description": "Combined ID"}
        #     ],
        # )
        # processed = processed.project(["product_id"])

        # output = processed.run(pz_config)
        # return output

    def _execute_q10(self) -> tuple[list[dict], dict]:
        """Q10: Based on product images, find matching outfits consisting of shoes, bottomwear, and topwear in Black, Blue, Red, or White, where all three items are from the same brand, same color, and each is priced at $10 or less."""
        # TODO: come back once carnot supports image data
        return [], {"total_tokens": 0, "total_execution_cost": 0.0}
        # def add_image_data(
        #     styles_details: pz.MemoryDataset, data_dir: str, suffix: str
        # ) -> pz.MemoryDataset:
        #     styles_details = styles_details.add_columns(
        #         udf=lambda row: {
        #             "image_file_path"
        #             + suffix: os.path.join(
        #                 data_dir,
        #                 "images",
        #                 str(row["prod_id" + suffix]) + ".jpg",
        #             )
        #         },
        #         cols=[
        #             {
        #                 "name": "image_file_path" + suffix,
        #                 "type": pz.ImageFilepath,
        #                 "description": "",  # leave empty because this influences the LLMs decision
        #             }
        #         ],
        #     )
        #     return styles_details


        # def run(pz_config, data_dir: str):
        #     # Load data
        #     styles_details = pd.read_parquet(
        #         os.path.join(data_dir, "styles_details.parquet")
        #     ).rename(
        #         columns={"id": "prod_id"}
        #     )  # prevent naming conflict with internal Palimpzest 'id' column

        #     # Pre-filter data
        #     styles_details = styles_details[
        #         (styles_details["baseColour"].isin(["Black", "Blue", "Red", "White"]))
        #         & (styles_details["price"] <= 1000)
        #     ]

        #     footwear = styles_details.copy(deep=True)
        #     footwear.columns = [col + "_footwear" for col in footwear.columns]
        #     footwear = pz.MemoryDataset(id="footwear", vals=footwear)
        #     footwear = add_image_data(footwear, data_dir, "_footwear")

        #     bottomwear = styles_details.copy(deep=True)
        #     bottomwear.columns = [col + "_bottomwear" for col in bottomwear.columns]
        #     bottomwear = pz.MemoryDataset(id="bottomwear", vals=bottomwear)
        #     bottomwear = add_image_data(bottomwear, data_dir, "_bottomwear")

        #     topwear = styles_details.copy(deep=True)
        #     topwear.columns = [col + "_topwear" for col in topwear.columns]
        #     topwear = pz.MemoryDataset(id="topwear", vals=topwear)
        #     topwear = add_image_data(topwear, data_dir, "_topwear")

        #     # Filters
        #     footwear_f = footwear.sem_filter(
        #         "The image depicts a (pair of) shoe(s), sandal(s), flip-flop(s). If there are multiple products in the picture, always refer to the most promiment one.",
        #         depends_on=["image_file_path_footwear"],
        #     )

        #     bottomwear_f = bottomwear.sem_filter(
        #         "The image depicts a piece of apparel that can be worn on the lower part of the body, like pants, shorts, skirts, ... If there are multiple products in the picture, always refer to the most promiment one.",
        #         depends_on=["image_file_path_bottomwear"],
        #     )

        #     topwear_f = topwear.sem_filter(
        #         "The image depicts a piece of apparel that can be worn on the upper part of the body, like t-shirts, shirts, pullovers, hoodies, but still require some sort of clothing on the lower body, which means, e.g., not a dress. If there are multiple products in the picture, always refer to the most promiment one.",
        #         depends_on=["image_file_path_topwear"],
        #     )

        #     # Perform joins
        #     processed_1 = footwear_f.sem_join(
        #         bottomwear_f,
        #         """
        #         The images depict products with the same primary base color, e.g., both are black, both are white, and both products are from the same brand.
        #         """,
        #         depends_on=[
        #             "productDisplayName_footwear",
        #             "productDescriptors_footwear",
        #             "image_file_path_footwear",
        #             "productDisplayName_bottomwear",
        #             "productDescriptors_bottomwear",
        #             "image_file_path_bottomwear",
        #         ],
        #     )

        #     processed_2 = processed_1.sem_join(
        #         topwear_f,
        #         """
        #         The images depict products with the same primary base color, e.g., both are black, both are white, and both products are from the same brand.
        #         """,
        #         depends_on=[
        #             "productDisplayName_bottomwear",
        #             "productDescriptors_bottomwear",
        #             "image_file_path_bottomwear",
        #             "productDisplayName_topwear",
        #             "productDescriptors_topwear",
        #             "image_file_path_topwear",
        #         ],
        #     )

        #     # Generate joined identifiers
        #     processed_3 = processed_2.add_columns(
        #         udf=lambda row: {
        #             "product_id": str(row["prod_id_footwear"])
        #             + "-"
        #             + str(row["prod_id_bottomwear"])
        #             + "-"
        #             + str(row["prod_id_topwear"])
        #         },
        #         cols=[
        #             {"name": "product_id", "type": str, "description": "Combined ID"}
        #         ],
        #     )
        #     processed_4 = processed_3.project(["product_id"])

        #     output = processed_4.run(pz_config)
        #     return output

    def _execute_q11(self) -> tuple[list[dict], dict]:
        """Q11: Based on product images and descriptions, find matching all-black outfits consisting of shoes, bottomwear (excluding swimwear), topwear (excluding swimwear), and an accessory (watch, jewellery, or bag priced at $5 or less), where all four items are from the same brand."""
        # TODO: come back once carnot supports image data
        return [], {"total_tokens": 0, "total_execution_cost": 0.0}

        # def add_image_data(
        #     styles_details: pz.MemoryDataset, data_dir: str, suffix: str
        # ) -> pz.MemoryDataset:
        #     styles_details = styles_details.add_columns(
        #         udf=lambda row: {
        #             "image_file_path"
        #             + suffix: os.path.join(
        #                 data_dir,
        #                 "images",
        #                 str(row["prod_id" + suffix]) + ".jpg",
        #             )
        #         },
        #         cols=[
        #             {
        #                 "name": "image_file_path" + suffix,
        #                 "type": pz.ImageFilepath,
        #                 "description": "",  # leave empty because this influences the LLMs decision
        #             }
        #         ],
        #     )
        #     return styles_details


        # def run(pz_config, data_dir: str):
        #     # Load data
        #     styles_details = pd.read_parquet(
        #         os.path.join(data_dir, "styles_details.parquet")
        #     ).rename(
        #         columns={"id": "prod_id"}
        #     )  # prevent naming conflict with internal Palimpzest 'id' column

        #     footwear = styles_details.copy(deep=True)
        #     footwear.columns = [col + "_footwear" for col in footwear.columns]
        #     footwear = pz.MemoryDataset(id="footwear", vals=footwear)
        #     footwear = add_image_data(footwear, data_dir, "_footwear")

        #     bottomwear = styles_details.copy(deep=True)
        #     bottomwear.columns = [col + "_bottomwear" for col in bottomwear.columns]
        #     bottomwear = pz.MemoryDataset(id="bottomwear", vals=bottomwear)
        #     bottomwear = add_image_data(bottomwear, data_dir, "_bottomwear")

        #     topwear = styles_details.copy(deep=True)
        #     topwear.columns = [col + "_topwear" for col in topwear.columns]
        #     topwear = pz.MemoryDataset(id="topwear", vals=topwear)
        #     topwear = add_image_data(topwear, data_dir, "_topwear")

        #     accessories = styles_details.copy(deep=True)
        #     accessories = accessories[accessories["price"] <= 500]
        #     accessories.columns = [col + "_accessories" for col in accessories.columns]
        #     accessories = pz.MemoryDataset(id="accessories", vals=accessories)
        #     accessories = add_image_data(accessories, data_dir, "_accessories")

        #     # Filters
        #     footwear_f = footwear.sem_filter("""
        #         You will receive an image and a description of a product.
        #         Determine whether the product can be worn on the feet, like shoes, sandals, flip-flops, ...
        #         The predominant color of the depicted product should be black.
        #         If there are multiple products in the picture, always refer to the most promiment one.
        #         The description of the product is as follows.""",
        #         depends_on=[
        #             "productDisplayName_footwear",
        #             "productDescriptors_footwear",
        #         ],
        #     )

        #     bottomwear_f = bottomwear.sem_filter("""
        #         You will receive an image and a description of a product.
        #         Determine whether the product can be worn on the lower part of the body, like pants, shorts, skirts, ...
        #         The predominant color of the depicted product should be black.
        #         Do not consider swimwear.
        #         If there are multiple products in the picture, always refer to the most promiment one.
        #         The description of the product is as follows.""",
        #         depends_on=[
        #             "productDisplayName_bottomwear",
        #             "productDescriptors_bottomwear",
        #         ],
        #     )

        #     topwear_f = topwear.sem_filter("""
        #         You will receive an image and a description of a product.
        #         Determine whether the product can be worn on the upper part of the body, like t-shirts, shirts, pullovers, hoodies, but still require some sort of clothing on the lower body, which means, e.g., not a dress.
        #         The predominant color of the depicted product should be black.
        #         Do not consider swimwear.
        #         If there are multiple products in the picture, always refer to the most promiment one.
        #         The description of the product is as follows.""",
        #         depends_on=[
        #             "productDisplayName_topwear",
        #             "productDescriptors_topwear",
        #         ],
        #     )

        #     accessories_f = accessories.sem_filter("""
        #         You will receive an image and a description of a product.
        #         Determine whether the product a watch or some jewellery or a bag.
        #         A bag might be a handbag or a (gym) backpack or some other type of bag.
        #         If there are multiple products in the picture, always refer to the most promiment one.
        #         The description of the product is as follows.""",
        #         depends_on=[
        #             "productDisplayName_accessories",
        #             "productDescriptors_accessories",
        #         ],
        #     )

        #     # Perform joins
        #     processed_1 = footwear_f.sem_join(
        #         accessories_f,
        #         """
        #         You will receive a description and an image of two products.
        #         Determine whether they are from the same brand.
        #         """,
        #         depends_on=[
        #             "productDisplayName_footwear",
        #             "productDescriptors_footwear",
        #             "image_file_path_footwear",
        #             "productDisplayName_accessories",
        #             "productDescriptors_accessories",
        #             "image_file_path_accessories",
        #         ],
        #     )

        #     processed_2 = topwear_f.sem_join(
        #         bottomwear_f,
        #         """
        #         You will receive a description and an image of two products.
        #         Determine whether they are from the same brand.
        #         """,
        #         depends_on=[
        #             "productDisplayName_topwear",
        #             "productDescriptors_topwear",
        #             "image_file_path_topwear",
        #             "productDisplayName_bottomwear",
        #             "productDescriptors_bottomwear",
        #             "image_file_path_bottomwear",
        #         ],
        #     )

        #     processed_3 = processed_1.sem_join(
        #         processed_2,
        #         """
        #         You will receive a description and an image of two products.
        #         Determine whether they are from the same brand.
        #         """,
        #         depends_on=[
        #             "productDisplayName_footwear",
        #             "productDescriptors_footwear",
        #             "image_file_path_footwear",
        #             "productDisplayName_topwear",
        #             "productDescriptors_topwear",
        #             "image_file_path_topwear",
        #         ],
        #     )

        #     # Generate joined identifiers
        #     processed_4 = processed_3.add_columns(
        #         udf=lambda row: {
        #             "product_id": str(row["prod_id_footwear"])
        #             + "-"
        #             + str(row["prod_id_bottomwear"])
        #             + "-"
        #             + str(row["prod_id_topwear"])
        #             + "-"
        #             + str(row["prod_id_accessories"])
        #         },
        #         cols=[
        #             {"name": "product_id", "type": str, "description": "Combined ID"}
        #         ],
        #     )
        #     processed_5 = processed_4.project(["product_id"])

        #     output = processed_5.run(pz_config)
        #     return output

    def _execute_q12(self) -> tuple[list[dict], dict]:
        """Q12: For each Adidas or Puma product, use the product image and description to generate a JSON object with the product id, brand name (lowercase), and master category classified as 'accessories', 'apparel', or 'footwear'."""
        # TODO: come back once carnot supports image data
        return [], {"total_tokens": 0, "total_execution_cost": 0.0}

        # def add_image_data(
        #     styles_details: pz.MemoryDataset, data_dir: str
        # ) -> pz.MemoryDataset:
        #     styles_details = styles_details.add_columns(
        #         udf=lambda row: {
        #             "image_file_path": os.path.join(
        #                 data_dir, "images", str(row["prod_id"]) + ".jpg"
        #             )
        #         },
        #         cols=[
        #             {
        #                 "name": "image_file_path",
        #                 "type": pz.ImageFilepath,
        #                 "description": "",  # leave empty because this influences the LLMs decision
        #             }
        #         ],
        #     )
        #     return styles_details


        # def run(pz_config, data_dir: str):
        #     # Load data
        #     styles_details = pd.read_parquet(
        #         os.path.join(data_dir, "styles_details.parquet")
        #     ).rename(
        #         columns={"id": "prod_id"}
        #     )  # prevent naming conflict with internal Palimpzest 'id' column

        #     styles_details = styles_details[styles_details.apply(lambda row: row['masterCategory']['typeName'] in ('Accessories', 'Apparel', 'Footwear'), axis=1)]
            
        #     styles_details = pz.MemoryDataset(id="styles_details", vals=styles_details)
        #     styles_details = add_image_data(styles_details, data_dir)

        #     # Filter data
        #     styles_details = styles_details.sem_filter(
        #         "Does the following description describe a product from either Adidas or Puma?",
        #         depends_on=[
        #             "productDisplayName",
        #             "productDescriptors",
        #         ],
        #     )

        #     # Project JSON
        #     styles_details = styles_details.sem_add_columns(
        #         cols=[
        #             {
        #                 "name": "product_id",
        #                 "type": str,
        #                 "description": """
        #                     You are given a product description and an image of the product as well as the product id.
        #                     The product contains a fashion item (clothing, shoes, accessories, etc).
        #                     There might be multiple fashion items in the image, especially when a model is presenting them.
        #                     If this is the case, focus only on the primary fashion item and use the description to determine which item in the image is of interest.

        #                     For each product, generate the following JSON:
        #                     ```
        #                     {
        #                     "id": <product id> (integer),
        #                     "brand": <extract the brand name from the description and/or image. use lower-case letters for the brand name>",
        #                     "category": <classify the images into 'accessories', 'apparel', 'footwear'>
        #                     }
        #                     ```

        #                     Output the json in a single line.
        #                     Keep the order of the keys in the JSON as given in the description.
        #                     Do not use spaces between { or keys and values in the JSON, i.e., do no use spaces anywhere in the JSON structure.
        #                     Use normal quotes in the JSON; do not use single quotes.
        #                 """
        #             }
        #         ],
        #         depends_on=["prod_id", "productDisplayName", "productDescriptors", "image_file_path"],
        #     )
            
        #     styles_details = styles_details.project(["product_id"])

        #     output = styles_details.run(pz_config)
        #     return output

    def _execute_q13(self) -> tuple[list[dict], dict]:
        """Q13: Based on product images and descriptions, find men's running shirts with round neck and short sleeves, in blue or black (not bright colors like white, and definitely not green), with a striped design, suitable for outdoor running in warm weather."""
        # def add_image_data(
        #     styles_details: pz.MemoryDataset, data_dir: str
        # ) -> pz.MemoryDataset:
        #     styles_details = styles_details.add_columns(
        #         udf=lambda row: {
        #             "image_file_path": os.path.join(
        #                 data_dir, "images", str(row["product_id"]) + ".jpg"
        #             )
        #         },
        #         cols=[
        #             {
        #                 "name": "image_file_path",
        #                 "type": pz.ImageFilepath,
        #                 "description": "",  # leave empty because this influences the LLMs decision
        #             }
        #         ],
        #     )
        #     return styles_details


        # def run(pz_config, data_dir: str):
        #     # Load data
        #     styles_details = pd.read_parquet(
        #         os.path.join(data_dir, "styles_details.parquet")
        #     ).rename(
        #         columns={"id": "product_id"}
        #     )  # prevent naming conflict with internal Palimpzest 'id' column
            
        #     styles_details = pz.MemoryDataset(id="styles_details", vals=styles_details)
        #     styles_details = add_image_data(styles_details, data_dir)

        #     # Filter data
        #     styles_details = styles_details.sem_filter(
        #         """
        #         You will receive a description of what a customer is looking for together with an image and a textual description of the product.
        #         Determine if they both match.
            
        #         I am looking for a running shirt for men with a round neck and short sleeves,
        #         preferably in blue or black, but not bright colors like white.
        #         Also definitely not green.
        #         It should be suitable for outdoor running in warm weather.
        #         If the t-shirt is not green, it should at least feature a striped design.

        #         The product has the following image and textual description:
        #         """,
        #         depends_on=[
        #             "productDisplayName",
        #             "productDescriptors",
        #             "image_file_path",
        #         ],
        #     )
            
        #     styles_details = styles_details.project(["product_id"])

        #     output = styles_details.run(pz_config)
        #     return output
