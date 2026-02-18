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

    # Filter data
    styles_details = styles_details.sem_filter(
        "The product is a backpack from Reebok",
        depends_on=["productDisplayName", "productDescriptors"],
    )
    styles_details = styles_details.project(["product_id"])

    output = styles_details.run(pz_config)
    return output
