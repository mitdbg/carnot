"""
Carnot system runner implementation.
"""

import os
import sys
from pathlib import Path

import carnot

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from runner.generic_carnot_runner.generic_carnot_runner import GenericCarnotRunner


class CarnotRunner(GenericCarnotRunner):
    """Runner for the Carnot system."""

    def __init__(
        self,
        use_case: str,
        scale_factor: int,
        model_name: str = "gemini-2.5-flash",
        concurrent_llm_worker=20,
        skip_setup: bool = False,
    ):
        """
        Initialize Carnot runner.

        Args:
            use_case: The use case to run
            model_name: LLM model to use
        """
        self.llm_config = {"OPENAI_API_KEY": os.getenv("OPENAI_API_KEY")}
        super().__init__(
            use_case,
            scale_factor,
            model_name,
            concurrent_llm_worker,
            skip_setup,
        )

    def _execute_q1(self) -> tuple[list[dict], dict]:
        """Q1: Who is the director of the movie that has Ben Piazza in the role of Bob Whitewood?"""
        table_df = self.load_data("ben_piazza.csv")
        text_df = self.load_data("ben_piazza_text_data.csv")
        joined_df = table_df.merge(
            text_df,
            left_on="Title",
            right_on="title",
            how="left",
        )
        joined_df = joined_df.fillna("")
        dataset = carnot.Dataset(
            name="Movie Data",
            annotation="",
            items=joined_df.to_dict(orient="records"),
        )
        execution = carnot.Execution(
            query="Who is the director of the movie that has Ben Piazza in the role of Bob Whitewood? Return the director's name as director.",
            datasets=[dataset],
            llm_config=self.llm_config,
        )
        _, plan = execution.plan()
        execution._plan = plan
        output, _ = execution.run()

        return output, {"total_tokens": 0, "total_execution_cost": 0.0}

    def _execute_q2a(self) -> tuple[list[dict], dict]:
        """Q2a: Identify the images containing logos, if available, for each racetrack in which A.P. Warrior was a contender."""
        # TODO: come back once carnot supports image data
        return [], {"total_tokens": 0, "total_execution_cost": 0.0}
        # pz_images = pz.ImageFileDataset(
        #     id="images", path=os.path.join(self.data_path, "images")
        # )
        # table_df = self.load_data("ap_warrior.csv")
        # pz_table = pz.MemoryDataset(id="ap_warrior_table", vals=table_df)

        # prompt = "You will be provided with a horse racetrack name and an image. Determine if the image shows the logo of the racetrack."  # noqa: E501
        # pz_table = pz_table.sem_join(
        #     pz_images,
        #     prompt,
        #     depends_on=[
        #         "Track",
        #         "contents",
        #     ],
        # )
        # pz_table = pz_table.project(["ID", "filename"])
        # output = pz_table.run(self.palimpzest_config())

        # return output

    def _execute_q2b(self) -> tuple[list[dict], dict]:
        """Q2b: Identify the images containing logos, if available, for each racetrack in which A.P. Warrior was a contender. What is the color of each logo?"""
        # TODO: come back once carnot supports image data
        return [], {"total_tokens": 0, "total_execution_cost": 0.0}
        # pz_images = pz.ImageFileDataset(
        #     id="images", path=os.path.join(self.data_path, "images")
        # )
        # table_df = self.load_data("ap_warrior.csv")
        # pz_table = pz.MemoryDataset(id="ap_warrior_table", vals=table_df)

        # prompt = "You will be provided with a horse racetrack name and an image. Determine if the image shows the logo of the racetrack."  # noqa: E501
        # pz_table = pz_table.sem_join(
        #     pz_images,
        #     prompt,
        #     depends_on=[
        #         "Track",
        #         "contents",
        #     ],
        # )

        # prompt = "The color of the logo in the image"
        # pz_table = pz_table.sem_map(
        #     [
        #         {
        #             "name": "color",
        #             "type": str,
        #             "desc": prompt,
        #         }
        #     ],
        #     depends_on=["contents"],
        # )
        # pz_table = pz_table.project(["ID", "filename", "color"])
        # output = pz_table.run(self.palimpzest_config())

        # return output

    def _execute_q3a(self) -> tuple[list[dict], dict]:
        """Q3a: Which movies are comedies?"""
        text_df = self.load_data("lizzy_caplan_text_data.csv")
        dataset = carnot.Dataset(
            name="Movie Data",
            annotation="",
            items=text_df.to_dict(orient="records"),
        )
        execution = carnot.Execution(
            query="Which movies are comedies? Return the title of each movie.",
            datasets=[dataset],
            llm_config=self.llm_config,
        )
        _, plan = execution.plan()
        execution._plan = plan
        output, _ = execution.run()

        return output, {"total_tokens": 0, "total_execution_cost": 0.0}

    def _execute_q3f(self) -> tuple[list[dict], dict]:
        """Q3f: Which movies are romantic comedies?"""
        text_df = self.load_data("lizzy_caplan_text_data.csv")
        dataset = carnot.Dataset(
            name="Movie Data",
            annotation="",
            items=text_df.to_dict(orient="records"),
        )
        execution = carnot.Execution(
            query="Which movies are romantic comedies? Return the title of each movie.",
            datasets=[dataset],
            llm_config=self.llm_config,
        )
        _, plan = execution.plan()
        execution._plan = plan
        output, _ = execution.run()

        return output, {"total_tokens": 0, "total_execution_cost": 0.0}

    def _execute_q4(self) -> dict:
        """Q4: Categorize the movies in the table by their genre. If a movie belongs to multiple genres, list it under each applicable genre"""
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
        dataset = carnot.Dataset(
            name="Movie Data",
            annotation="",
            items=text_df.to_dict(orient="records"),
        )
        execution = carnot.Execution(
            query="Categorize the movies in the table by their genre. If a movie belongs to multiple genres, list it under each applicable genre. Return genre and movies_in_genre, where movies_in_genre contains a comma-separated list of movie titles that belong to that genre.",
            datasets=[dataset],
            llm_config=self.llm_config,
        )
        _, plan = execution.plan()
        execution._plan = plan
        output, _ = execution.run()

        return output, {"total_tokens": 0, "total_execution_cost": 0.0}

    def _execute_q5(self):
        """Q5: Who has played a role in all the following movies: 'Love Is the Drug', 'Crashing', 'Cloverfield', 'My Best Friend's Girl', 'Hot Tub Time Machine', 'The Last Rites of Ransom Pride', 'Save the Date', 'Bachelorette', '3, 2, 1... Frankie Go Boom', 'Queens of Country', 'Item 47', 'The Night Before', 'Now You See Me 2', 'Allied', 'Extinction', and 'Cobweb'?"""
        text_df = self.load_data("lizzy_caplan_text_data.csv", sep=",", quotechar='"')
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
        dataset = carnot.Dataset(
            name="Movie Data",
            annotation="",
            items=text_df.to_dict(orient="records"),
        )
        execution = carnot.Execution(
            query="Who has played a role in all the following movies: 'Love Is the Drug', 'Crashing', 'Cloverfield', 'My Best Friend's Girl', 'Hot Tub Time Machine', 'The Last Rites of Ransom Pride', 'Save the Date', 'Bachelorette', '3, 2, 1... Frankie Go Boom', 'Queens of Country', 'Item 47', 'The Night Before', 'Now You See Me 2', 'Allied', 'Extinction', and 'Cobweb'? Return the name of the actor as actor.",
            datasets=[dataset],
            llm_config=self.llm_config,
        )
        _, plan = execution.plan()
        execution._plan = plan
        output, _ = execution.run()

        return output, {"total_tokens": 0, "total_execution_cost": 0.0}

    def _execute_q6a(self) -> tuple[list[dict], dict]:
        """Q6a: Which airlines have destinations in Frankfurt?"""
        table_df = self.load_data("tampa_international_airport.csv")
        dataset = carnot.Dataset(
            name="Airline Data",
            annotation="",
            items=table_df.to_dict(orient="records"),
        )
        execution = carnot.Execution(
            query="Which airlines have destinations in Frankfurt? Return the airline names as Airlines.",
            datasets=[dataset],
            llm_config=self.llm_config,
        )
        _, plan = execution.plan()
        execution._plan = plan
        output, _ = execution.run()

        return output, {"total_tokens": 0, "total_execution_cost": 0.0}

    def _execute_q6b(self) -> tuple[list[dict], dict]:
        """Q6b: Which airlines have destinations in Germany?"""
        table_df = self.load_data("tampa_international_airport.csv")
        dataset = carnot.Dataset(
            name="Airline Data",
            annotation="",
            items=table_df.to_dict(orient="records"),
        )
        execution = carnot.Execution(
            query="Which airlines have destinations in Germany? Return the airline names as Airlines.",
            datasets=[dataset],
            llm_config=self.llm_config,
        )
        _, plan = execution.plan()
        execution._plan = plan
        output, _ = execution.run()

        return output, {"total_tokens": 0, "total_execution_cost": 0.0}

    def _execute_q6c(self) -> tuple[list[dict], dict]:
        """Q6c: Which airlines have destinations in Europe?"""
        table_df = self.load_data("tampa_international_airport.csv")
        dataset = carnot.Dataset(
            name="Airline Data",
            annotation="",
            items=table_df.to_dict(orient="records"),
        )
        execution = carnot.Execution(
            query="Which airlines have destinations in Europe? Return the airline names as Airlines.",
            datasets=[dataset],
            llm_config=self.llm_config,
        )
        _, plan = execution.plan()
        execution._plan = plan
        output, _ = execution.run()

        return output, {"total_tokens": 0, "total_execution_cost": 0.0}

    def _execute_q7(self) -> tuple[list[dict], dict]:
        """Q7: For each airline with destinations in Europe, find its logo if one exists."""
        # TODO: come back once carnot supports image data
        return [], {"total_tokens": 0, "total_execution_cost": 0.0}
        # pz_images = pz.ImageFileDataset(
        #     id="images", path=os.path.join(self.data_path, "images")
        # )
        # table_df = self.load_data("tampa_international_airport.csv")
        # pz_table = pz.MemoryDataset(id="tampa_airport", vals=table_df)

        # prompt = "The image shows the airline logo."
        # pz_table = pz_table.sem_join(
        #     pz_images,
        #     prompt,
        #     depends_on=[
        #         "Airlines",
        #         "contents",
        #     ],
        # )
        # pz_table = pz_table.project(["Airlines", "filename"])
        # output = pz_table.run(self.palimpzest_config())

        # return output
