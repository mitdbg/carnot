"""
Palimpzest system runner implementation.
"""

import os

import palimpzest as pz
import pandas as pd
from palimpzest.core.elements.records import DataRecordCollection

from src.runner.generic_palimpzest_runner.generic_palimpzest_runner import (
    GenericPalimpzestRunner,
)


class PalimpzestRunner(GenericPalimpzestRunner):
    """Runner for the Palimpzest system."""

    def __init__(
        self,
        use_case: str,
        scale_factor: int,
        model_name: str = "gemini-2.5-flash",
        concurrent_llm_worker=20,
        skip_setup: bool = False,
    ):
        """
        Initialize Palimpzest runner.

        Args:
            use_case: The use case to run
            model_name: LLM model to use
        """

        super().__init__(
            use_case,
            scale_factor,
            model_name,
            concurrent_llm_worker,
            skip_setup,
        )

    def _execute_q1(self) -> DataRecordCollection:
        table_df = self.load_data("ben_piazza.csv")
        text_df = self.load_data("ben_piazza_text_data.csv")
        joined_df = table_df.merge(
            text_df,
            left_on="Title",
            right_on="title",
            how="left",
        )
        joined_df = joined_df.fillna("")
        pz_dataset = pz.MemoryDataset(id="ben_piazza_joined", vals=joined_df)

        prompt = "Extract the director name from the movie description."
        pz_dataset = pz_dataset.sem_map(
            [
                {
                    "name": "director",
                    "type": str,
                    "desc": prompt,
                }
            ],
            depends_on=["text"],
        )
        pz_dataset = pz_dataset.filter(
            lambda row: row["Role"] == "Bob Whitewood"
        )
        pz_dataset = pz_dataset.project(["director"])
        output = pz_dataset.run(self.palimpzest_config())

        return output

    def _execute_q2a(self) -> DataRecordCollection:
        pz_images = pz.ImageFileDataset(
            id="images", path=os.path.join(self.data_path, "images")
        )
        table_df = self.load_data("ap_warrior.csv")
        pz_table = pz.MemoryDataset(id="ap_warrior_table", vals=table_df)

        prompt = "You will be provided with a horse racetrack name and an image. Determine if the image shows the logo of the racetrack."  # noqa: E501
        pz_table = pz_table.sem_join(
            pz_images,
            prompt,
            depends_on=[
                "Track",
                "contents",
            ],
        )
        pz_table = pz_table.project(["ID", "filename"])
        output = pz_table.run(self.palimpzest_config())

        return output

    def _execute_q2b(self) -> DataRecordCollection:
        pz_images = pz.ImageFileDataset(
            id="images", path=os.path.join(self.data_path, "images")
        )
        table_df = self.load_data("ap_warrior.csv")
        pz_table = pz.MemoryDataset(id="ap_warrior_table", vals=table_df)

        prompt = "You will be provided with a horse racetrack name and an image. Determine if the image shows the logo of the racetrack."  # noqa: E501
        pz_table = pz_table.sem_join(
            pz_images,
            prompt,
            depends_on=[
                "Track",
                "contents",
            ],
        )

        prompt = "The color of the logo in the image"
        pz_table = pz_table.sem_map(
            [
                {
                    "name": "color",
                    "type": str,
                    "desc": prompt,
                }
            ],
            depends_on=["contents"],
        )
        pz_table = pz_table.project(["ID", "filename", "color"])
        output = pz_table.run(self.palimpzest_config())

        return output

    def _execute_q3a(self) -> DataRecordCollection:
        text_df = self.load_data("lizzy_caplan_text_data.csv")
        pz_text = pz.MemoryDataset(id="lizzy_caplan_text", vals=text_df)

        prompt = (
            "Determine if a movie is a comedy movie given their description."
        )
        pz_text = pz_text.sem_filter(prompt, depends_on=["title", "text"])
        pz_text = pz_text.project(["title"])
        output = pz_text.run(self.palimpzest_config())

        return output

    def _execute_q3f(self) -> DataRecordCollection:
        text_df = self.load_data("lizzy_caplan_text_data.csv")
        pz_text = pz.MemoryDataset(id="lizzy_caplan_text", vals=text_df)

        prompt = (
            "Determine if a movie is a romantic comedy given their description."
        )
        pz_text = pz_text.sem_filter(prompt, depends_on=["title", "text"])
        pz_text = pz_text.project(["title"])
        output = pz_text.run(self.palimpzest_config())

        return output

    def _execute_q4(self) -> dict:
        text_df = self.load_data("lizzy_caplan_text_data.csv")
        target_values = [
            "Orange County",
            "Mean Girls",
            "Love Is the Drug",
            "Crashing",
            "Cloverfield",
            "My Best Friend's Girl",
            "Crossing Over",
            "Hot Tub Time Machine",
            "The Last Rites of Ransom Pride",
            "127 Hours",
            "High Road",
            "Save the Date",
            "Bachelorette",
            "3, 2, 1... Frankie Go Boom",
            "Queens of Country",
            "Item 47",
            "The Interview",
            "The Night Before",
            "Now You See Me 2",
            "Allied",
            "The Disaster Artist",
            "Extinction",
            "The People We Hate at the Wedding",
            "Cobweb",
        ]
        text_df = text_df[text_df["title"].isin(target_values)]
        pz_text = pz.MemoryDataset(id="lizzy_caplan_text", vals=text_df)

        pz_text = pz_text.sem_map(
            [
                {
                    "name": "genres",
                    "type": str,
                    "desc": "The genres of the movie, separated by commas",
                }
            ],
            depends_on=["text"],
        )
        pz_text = pz_text.project(["title", "genres"])
        output = pz_text.run(self.palimpzest_config())
        output_df = output.to_df()

        expanded_data = []
        for _, row in output_df.iterrows():
            movie_title = row["title"]
            genres = []
            if isinstance(row["genres"], str):
                genres = [
                    genre.lower().strip() for genre in row["genres"].split(",")
                ]

            for genre in genres:
                expanded_data.append({"genre": genre, "title": movie_title})

        df_expanded = pd.DataFrame(expanded_data)
        genre_movies_table = (
            df_expanded.groupby("genre")["title"]
            .apply(lambda x: ", ".join(x))
            .reset_index()
        )
        genre_movies_table.rename(
            columns={"title": "movies_in_genre"}, inplace=True
        )

        return {
            "results": genre_movies_table,
            "execution_stats": output.execution_stats,
        }

    def _execute_q5(self):
        text_df = self.load_data(
            "lizzy_caplan_text_data.csv", sep=",", quotechar='"'
        )
        target_values = [
            "Love Is the Drug",
            "Crashing",
            "Cloverfield",
            "My Best Friend's Girl",
            "Hot Tub Time Machine",
            "The Last Rites of Ransom Pride",
            "Save the Date",
            "Bachelorette",
            "3, 2, 1... Frankie Go Boom",
            "Queens of Country",
            "Item 47",
            "The Night Before",
            "Now You See Me 2",
            "Allied",
            "Extinction",
            "Cobweb",
        ]
        text_df = text_df[text_df["title"].isin(target_values)]
        pz_text = pz.MemoryDataset(id="lizzy_caplan_text", vals=text_df)

        prompt = "Who has played a role in all the movies listed in the table given their descriptions? Simply give the name of the actor."  # noqa: E501
        pz_text = pz_text.sem_agg(
            col={
                "name": "actor",
                "type": str,
                "description": "The name of the actor",
            },
            agg=prompt,
            depends_on=["title", "text"],
        )
        output = pz_text.run(self.palimpzest_config())

        return output

    def _execute_q6a(self) -> DataRecordCollection:
        table_df = self.load_data("tampa_international_airport.csv")
        pz_table = pz.MemoryDataset(id="tampa_airport", vals=table_df)

        prompt = "Given destinations of an airline, the airline has flights to Frankfurt."  # noqa: E501
        pz_table = pz_table.sem_filter(
            prompt, depends_on=["Airlines", "Destinations"]
        )
        pz_table = pz_table.project(["Airlines"])
        output = pz_table.run(self.palimpzest_config())

        return output

    def _execute_q6b(self) -> DataRecordCollection:
        table_df = self.load_data("tampa_international_airport.csv")
        pz_table = pz.MemoryDataset(id="tampa_airport", vals=table_df)

        prompt = "Given destinations of an airline, the airline has flights to Germany."  # noqa: E501
        pz_table = pz_table.sem_filter(
            prompt, depends_on=["Airlines", "Destinations"]
        )
        pz_table = pz_table.project(["Airlines"])
        output = pz_table.run(self.palimpzest_config())

        return output

    def _execute_q6c(self) -> DataRecordCollection:
        table_df = self.load_data("tampa_international_airport.csv")
        pz_table = pz.MemoryDataset(id="tampa_airport", vals=table_df)

        prompt = "Given destinations of an airline, the airline has flights to Europe."  # noqa: E501
        pz_table = pz_table.sem_filter(
            prompt, depends_on=["Airlines", "Destinations"]
        )
        pz_table = pz_table.project(["Airlines"])
        output = pz_table.run(self.palimpzest_config())

        return output

    def _execute_q7(self) -> DataRecordCollection:
        pz_images = pz.ImageFileDataset(
            id="images", path=os.path.join(self.data_path, "images")
        )
        table_df = self.load_data("tampa_international_airport.csv")
        pz_table = pz.MemoryDataset(id="tampa_airport", vals=table_df)

        prompt = "The image shows the airline logo."
        pz_table = pz_table.sem_join(
            pz_images,
            prompt,
            depends_on=[
                "Airlines",
                "contents",
            ],
        )
        pz_table = pz_table.project(["Airlines", "filename"])
        output = pz_table.run(self.palimpzest_config())

        return output
