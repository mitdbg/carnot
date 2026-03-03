"""
Created on July 29

Carnot system runner implementation.
"""
import os
import sys
import textwrap
from pathlib import Path

import carnot

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from runner.generic_carnot_runner.generic_carnot_runner import GenericCarnotRunner


# TODO test on natural language queries from files/movie/query/natural_language
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
        """
        Execute Q1: "Find five clearly positive reviews (any movie)"

        Returns:
            DataFrame with columns: reviewId
        """
        reviews_df = self.load_data("Reviews.csv")
        dataset = carnot.Dataset(
            name="Reviews",
            annotation="",
            items=reviews_df.to_dict(orient="records"),
        )
        execution = carnot.Execution(
            query="Find five clearly positive reviews (any movie). Return the reviewId of these reviews.",
            datasets=[dataset],
            llm_config=self.llm_config,
        )
        _, plan = execution.plan()
        execution._plan = plan
        output, _ = execution.run()

        return output, {"total_tokens": 0, "total_execution_cost": 0.0}

    def _execute_q2(self) -> tuple[list[dict], dict]:
        """
        Execute Q2: "Find five positive reviews for movie "taken_3""

        Returns:
            DataFrame with columns: reviewId
        """
        reviews_df = self.load_data("Reviews.csv")
        reviews_df = reviews_df[reviews_df.id == "taken_3"]
        dataset = carnot.Dataset(
            name="Reviews",
            annotation="",
            items=reviews_df.to_dict(orient="records"),
        )
        execution = carnot.Execution(
            query='Find five clearly positive reviews for movie "taken_3". Return the reviewId of these reviews.',
            datasets=[dataset],
            llm_config=self.llm_config,
        )
        _, plan = execution.plan()
        execution._plan = plan
        output, _ = execution.run()

        return output, {"total_tokens": 0, "total_execution_cost": 0.0}

    def _execute_q3(self) -> tuple[list[dict], dict]:
        """
        Execute Q3: "Count of positive reviews for a "bad" movie (taken_3)"

        Returns:
            DataFrame with columns: positive_review_cnt
        """
        reviews_df = self.load_data("Reviews.csv")
        reviews_df = reviews_df[reviews_df.id == "taken_3"]
        dataset = carnot.Dataset(
            name="Reviews",
            annotation="",
            items=reviews_df.to_dict(orient="records"),
        )
        execution = carnot.Execution(
            query='Count the number of clearly positive reviews for movie "taken_3". Return the count as positive_review_cnt.',
            datasets=[dataset],
            llm_config=self.llm_config,
        )
        _, plan = execution.plan()
        execution._plan = plan
        output, _ = execution.run()

        return output, {"total_tokens": 0, "total_execution_cost": 0.0}

    def _execute_q4(self) -> tuple[list[dict], dict]:
        """
        Execute Q4: "Positivity ratio (average of 0/1) for a "bad" movie (taken_3)."

        Returns:
            DataFrame with columns: positivity_ratio
        """
        reviews_df = self.load_data("Reviews.csv")
        reviews_df = reviews_df[reviews_df.id == "taken_3"]
        dataset = carnot.Dataset(
            name="Reviews",
            annotation="",
            items=reviews_df.to_dict(orient="records"),
        )
        execution = carnot.Execution(
            query='Compute the average positivity (1 for clearly positive, 0 otherwise) for movie "taken_3". Return as positivity_ratio.',
            datasets=[dataset],
            llm_config=self.llm_config,
        )
        _, plan = execution.plan()
        execution._plan = plan
        output, _ = execution.run()

        return output, {"total_tokens": 0, "total_execution_cost": 0.0}

    def _execute_q5(self) -> tuple[list[dict], dict]:
        """
        Execute Q5: "Find pairs of reviews with same sentiment for the same movie."

        Returns:
            DataFrame with columns: id, reviewId, reviewId_right
        """
        reviews_df = self.load_data("Reviews.csv")
        reviews_df = reviews_df[reviews_df.id == "ant_man_and_the_wasp_quantumania"]
        dataset1 = carnot.Dataset(
            name="Reviews1",
            annotation="",
            items=reviews_df.to_dict(orient="records"),
        )
        dataset2 = carnot.Dataset(
            name="Reviews2",
            annotation="",
            items=reviews_df.rename(
                columns={col: f"{col}_right" for col in reviews_df.columns}
            ).to_dict(orient="records"),
        )
        execution = carnot.Execution(
            query="Find pairs of reviews for the same movie that express the same sentiment (both positive or both negative). Return id, reviewId, and reviewId_right. Limit to 10 pairs.",
            datasets=[dataset1, dataset2],
            llm_config=self.llm_config,
        )
        _, plan = execution.plan()
        execution._plan = plan
        output, _ = execution.run()

        return output, {"total_tokens": 0, "total_execution_cost": 0.0}

    def _execute_q6(self) -> tuple[list[dict], dict]:
        """
        Execute Q6: "Pairs of reviews that express the *opposite* sentiment for movie with id 'ant_man_and_the_wasp_quantumania'"

        Returns:
            DataFrame with columns: id, reviewId, reviewId_right
        """
        reviews_df = self.load_data("Reviews.csv")
        reviews_df = reviews_df[reviews_df.id == "ant_man_and_the_wasp_quantumania"]
        dataset1 = carnot.Dataset(
            name="Reviews1",
            annotation="",
            items=reviews_df.to_dict(orient="records"),
        )
        dataset2 = carnot.Dataset(
            name="Reviews2",
            annotation="",
            items=reviews_df.rename(
                columns={col: f"{col}_right" for col in reviews_df.columns}
            ).to_dict(orient="records"),
        )
        execution = carnot.Execution(
            query='Find pairs of reviews for movie "ant_man_and_the_wasp_quantumania" that express opposite sentiment (one positive, one negative). Return id, reviewId, and reviewId_right. Limit to 10 pairs.',
            datasets=[dataset1, dataset2],
            llm_config=self.llm_config,
        )
        _, plan = execution.plan()
        execution._plan = plan
        output, _ = execution.run()

        return output, {"total_tokens": 0, "total_execution_cost": 0.0}

    def _execute_q7(self) -> tuple[list[dict], dict]:
        """
        Execute Q7: All Pairs of reviews that express the *opposite* sentiment for movie with id 'ant_man_and_the_wasp_quantumania'

        Returns:
            DataFrame with columns: id, reviewId, reviewId_right
        """
        reviews_df = self.load_data("Reviews.csv")
        reviews_df = reviews_df[reviews_df.id == "ant_man_and_the_wasp_quantumania"]
        dataset1 = carnot.Dataset(
            name="Reviews1",
            annotation="",
            items=reviews_df.to_dict(orient="records"),
        )
        dataset2 = carnot.Dataset(
            name="Reviews2",
            annotation="",
            items=reviews_df.rename(
                columns={col: f"{col}_right" for col in reviews_df.columns}
            ).to_dict(orient="records"),
        )

        execution = carnot.Execution(
            query='Find all pairs of reviews for movie "ant_man_and_the_wasp_quantumania" that express opposite sentiment (one positive, one negative). Return id, reviewId, and reviewId_right.',
            datasets=[dataset1, dataset2],
            llm_config=self.llm_config,
        )
        _, plan = execution.plan()
        execution._plan = plan
        output, _ = execution.run()

        return output, {"total_tokens": 0, "total_execution_cost": 0.0}

    def _execute_q8(self) -> tuple[list[dict], dict]:
        """
        Execute Q8: "Calculate the number of positive and negative reviews for movie taken_3"

        Returns:
            DataFrame with columns: sentiment, count(sentiment)
        """
        reviews_df = self.load_data("Reviews.csv")
        reviews_df = reviews_df[reviews_df.id == "taken_3"]
        dataset = carnot.Dataset(
            name="Reviews",
            annotation="",
            items=reviews_df.to_dict(orient="records"),
        )
        execution = carnot.Execution(
            query='For movie "taken_3", classify each review as POSITIVE or NEGATIVE, then return the count for each sentiment as sentiment and count.',
            datasets=[dataset],
            llm_config=self.llm_config,
        )
        _, plan = execution.plan()
        execution._plan = plan
        output, _ = execution.run()

        return output, {"total_tokens": 0, "total_execution_cost": 0.0}

    def _execute_q9(self) -> tuple[list[dict], dict]:
        """
        Execute Q9: Score from 1 to 5 how much did the reviewer like the movie based on the movie reviews for movie 'ant_man_and_the_wasp_quantumania'.

        Returns:
            DataFrame with columns: reviewId, reviewScore
        """
        reviews_df = self.load_data("Reviews.csv")
        reviews_df = reviews_df[reviews_df.id == "ant_man_and_the_wasp_quantumania"]
        dataset = carnot.Dataset(
            name="Reviews",
            annotation="",
            items=reviews_df.to_dict(orient="records"),
        )
        execution = carnot.Execution(
            query=textwrap.dedent(
                """
                For movie "ant_man_and_the_wasp_quantumania", score each review from 1 to 5 based on how much the reviewer liked the movie. Use the following rubric to determine the score:

                Rubrics:
                5: Very positive. Strong positive sentiment, indicating high satisfaction.
                4: Positive. Noticeably positive sentiment, indicating general satisfaction.
                3: Neutral. Expresses no clear positive or negative sentiment. May be factual or descriptive without emotional language.
                2: Negative. Noticeably negative sentiment, indicating some level of dissatisfaction but without strong anger or frustration.
                1: Very negative. Strong negative sentiment, indicating high dissatisfaction, frustration, or anger.

                Return reviewId and reviewScore.
                """
            ),
            datasets=[dataset],
            llm_config=self.llm_config,
        )
        _, plan = execution.plan()
        execution._plan = plan
        output, _ = execution.run()

        return output, {"total_tokens": 0, "total_execution_cost": 0.0}

    def _execute_q10(self) -> tuple[list[dict], dict]:
        """
        Execute Q10: Rank the movies based on movie reviews. For each movie, score every review of it from 1 to 5, then calculate the average score of these reviews for each movie.

        Returns:
            DataFrame with columns: id, movieScore
        """
        reviews_df = self.load_data("Reviews.csv")
        dataset = carnot.Dataset(
            name="Reviews",
            annotation="",
            items=reviews_df.to_dict(orient="records"),
        )
        execution = carnot.Execution(
            query=textwrap.dedent(
                """
                For each movie, score every review from 1 to 5 based on how much the reviewer liked the movie. Use the following rubric to determine the score:

                Rubrics:
                5: Very positive. Strong positive sentiment, indicating high satisfaction.
                4: Positive. Noticeably positive sentiment, indicating general satisfaction.
                3: Neutral. Expresses no clear positive or negative sentiment. May be factual or descriptive without emotional language.
                2: Negative. Noticeably negative sentiment, indicating some level of dissatisfaction but without strong anger or frustration.
                1: Very negative. Strong negative sentiment, indicating high dissatisfaction, frustration, or anger.

                Then compute the average reviewScore per movie. Return id and movieScore (average score).
                """
            ),
            datasets=[dataset],
            llm_config=self.llm_config,
        )
        _, plan = execution.plan()
        execution._plan = plan
        output, _ = execution.run()

        return output, {"total_tokens": 0, "total_execution_cost": 0.0}
