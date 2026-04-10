import os
import pandas as pd
import palimpzest as pz


def run(pz_config, data_dir: str):
    # Load data
    styles_details = pd.read_parquet(
        os.path.join(data_dir, "styles_details.parquet")
    ).rename(
        columns={"id": "product_id"}
    )  # prevent naming conflict with internal Palimpzest 'id' column
    styles_details = pz.MemoryDataset(id="styles_details", vals=styles_details)

    # Perform map/extract
    styles_details = styles_details.sem_add_columns(
        cols=[
            {
                "name": "category",
                "type": str,
                "description": "Extract the brand name from the following product description. Only return the brand name, nothing else.",
            }
        ],
        depends_on=["productDisplayName", "productDescriptors"],
    )
    styles_details = styles_details.project(["product_id", "category"])

    output = styles_details.run(pz_config)
    return output
