"""
Created on July 29

Palimpzest system runner implementation.
"""

import pandas as pd
import palimpzest as pz
from palimpzest.core.elements.records import DataRecordCollection
from palimpzest.core.elements.groupbysig import GroupBySig
from pathlib import Path

# Add parent directory to path for imports
import sys

sys.path.append(str(Path(__file__).parent.parent.parent.parent))
from runner.generic_palimpzest_runner.generic_palimpzest_runner import (
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
        """
        Execute Q1: "Find five clearly positive reviews (any movie)"

        Returns:
            DataFrame with columns: reviewId
        """

        # dev branch
        reviews = pz.MemoryDataset(
            id="reviews", vals=self.load_data("Reviews.csv")
        )
        reviews = reviews.sem_filter(
            "Determine if the following movie review is clearly positive.",
            depends_on=["reviewText"],
        )
        reviews = reviews.project(["reviewId"])
        reviews = reviews.limit(5)

        output = reviews.run(self.palimpzest_config())

        return output

    def _execute_q2(self) -> DataRecordCollection:
        """
        Execute Q2: "Find five positive reviews for movie "taken_3""

        Returns:
            DataFrame with columns: reviewId
        """
        # dev branch
        reviews = pz.MemoryDataset(
            id="reviews",
            vals=self.load_data("Reviews.csv").rename(
                columns={"id": "movieId"}
            ),
        )
        reviews = reviews.filter(lambda r: r["movieId"] == "taken_3")
        reviews = reviews.sem_filter(
            "Determine if the following movie review is clearly positive.",
            depends_on=["reviewText"],
        )
        reviews = reviews.project(["reviewId"])
        reviews = reviews.limit(5)

        output = reviews.run(self.palimpzest_config())

        return output

    def _execute_q3(self) -> DataRecordCollection:
        """
        Execute Q3: "Count of positive reviews for a "bad" movie (taken_3)"

        Returns:
            DataFrame with columns: positive_review_cnt
        """
        # dev branch
        reviews = pz.MemoryDataset(
            id="reviews",
            vals=self.load_data("Reviews.csv").rename(
                columns={"id": "movieId"}
            ),
        )
        reviews = reviews.filter(lambda r: r["movieId"] == "taken_3")
        reviews = reviews.sem_filter(
            "Determine if the following movie review is clearly positive.",
            depends_on=["reviewText"],
        )
        reviews = reviews.count()

        output = reviews.run(self.palimpzest_config())

        return output

    def _execute_q4(self) -> DataRecordCollection:
        """
        Execute Q4: "Positivity ratio (average of 0/1) for a "bad" movie (taken_3)."

        Returns:
            DataFrame with columns: positivity_ratio
        """
        # dev branch
        reviews = pz.MemoryDataset(
            id="reviews",
            vals=self.load_data("Reviews.csv").rename(
                columns={"id": "movieId"}
            ),
        )
        reviews = reviews.filter(lambda r: r["movieId"] == "taken_3")
        reviews = reviews.sem_add_columns(
            [
                {
                    "name": "positivity",
                    "type": int,
                    "desc": "Return 1 if the following review is positive, and 0 if the review is not positive. Only output a single numeric value (1 or 0) with no additional commentary",
                }
            ],
            depends_on=["reviewText"],
        )
        reviews = reviews.project(["positivity"])
        reviews = reviews.average()

        output = reviews.run(self.palimpzest_config())

        return output

    def _execute_q5(self) -> DataRecordCollection:
        """
        Execute Q5: "Find pairs of reviews with same sentiment for the same movie."

        Returns:
            DataFrame with columns: id, reviewId_left, reviewId_right
        """
        reviews_df = self.load_data("Reviews.csv").rename(
            columns={"id": "movieId"}
        )

        input_df1 = reviews_df[
            reviews_df["movieId"] == "ant_man_and_the_wasp_quantumania"
        ]
        input_df2 = reviews_df[
            reviews_df["movieId"] == "ant_man_and_the_wasp_quantumania"
        ]
        input_df2 = input_df2.rename(
            columns={col: f"{col}_right" for col in input_df2.columns}
        )

        input1 = pz.MemoryDataset(id="input1", vals=input_df1)
        input2 = pz.MemoryDataset(id="input2", vals=input_df2)

        input3 = input1.sem_join(
            input2,
            condition="These two movie reviews express the same sentiment - either both are positive or both are negative.",
            depends_on=["reviewText", "reviewText_right"],
        )
        input3 = input3.project(["movieId", "reviewId", "reviewId_right"])
        input3 = input3.limit(10)

        output = input3.run(self.palimpzest_config())

        return output

    def _execute_q6(self) -> DataRecordCollection:
        """
        Execute Q6: "Pairs of reviews that express the *opposite* sentiment for movie with id 'ant_man_and_the_wasp_quantumania'"

        Returns:
            DataFrame with columns: id, reviewId_left, reviewId_right
        """

        reviews_df = self.load_data("Reviews.csv").rename(
            columns={"id": "movieId"}
        )

        input_df1 = reviews_df[
            reviews_df["movieId"] == "ant_man_and_the_wasp_quantumania"
        ]
        input_df2 = reviews_df[
            reviews_df["movieId"] == "ant_man_and_the_wasp_quantumania"
        ]
        input_df2 = input_df2.rename(
            columns={col: f"{col}_right" for col in input_df2.columns}
        )

        input1 = pz.MemoryDataset(id="input1", vals=input_df1)
        input2 = pz.MemoryDataset(id="input2", vals=input_df2)

        input3 = input1.sem_join(
            input2,
            condition="These two movie reviews express opposite sentiments - one is positive and the other is negative.",
            depends_on=["reviewText", "reviewText_right"],
        )
        input3 = input3.project(["movieId", "reviewId", "reviewId_right"])
        input3 = input3.limit(10)

        output = input3.run(self.palimpzest_config())

        return output

    def _execute_q7(self) -> DataRecordCollection:
        """
        Execute Q7: All Pairs of reviews that express the *opposite* sentiment for movie with id 'ant_man_and_the_wasp_quantumania'

        Returns:
            DataFrame with columns: movieId, reviewId_left, reviewId_right
        """

        reviews_df = self.load_data("Reviews.csv").rename(
            columns={"id": "movieId"}
        )

        input_df1 = reviews_df[
            reviews_df["movieId"] == "ant_man_and_the_wasp_quantumania"
        ]
        input_df2 = reviews_df[
            reviews_df["movieId"] == "ant_man_and_the_wasp_quantumania"
        ]
        input_df2 = input_df2.rename(
            columns={col: f"{col}_right" for col in input_df2.columns}
        )

        input1 = pz.MemoryDataset(id="input1", vals=input_df1)
        input2 = pz.MemoryDataset(id="input2", vals=input_df2)

        input3 = input1.sem_join(
            input2,
            condition="These two movie reviews express opposite sentiments - one is positive and the other is negative.",
            depends_on=["reviewText", "reviewText_right"],
        )
        input3 = input3.project(["movieId", "reviewId", "reviewId_right"])

        output = input3.run(self.palimpzest_config())

        return output

    def _execute_q8(self) -> DataRecordCollection:
        """
        Execute Q8: "Calculate the number of positive and negative reviews for movie taken_3"

        Returns:
            DataFrame with columns: sentiment, count(sentiment)
        """
        reviews = pz.MemoryDataset(
            id="reviews",
            vals=self.load_data("Reviews.csv").rename(
                columns={"id": "movieId"}
            ),
        )
        reviews = reviews.filter(lambda r: r["movieId"] == "taken_3")
        reviews = reviews.sem_add_columns(
            [
                {
                    "name": "sentiment",
                    "type": str,
                    "desc": "Return POSITIVE if the following review is positive, and NEGATIVE if the review is not positive. Only output POSITIVE or NEGATIVE with no additional commentary",
                }
            ],
            depends_on=["reviewText"],
        )
        reviews = reviews.project(["sentiment"])
        gby_desc = GroupBySig(
            group_by_fields=["sentiment"],
            agg_funcs=["count"],
            agg_fields=["sentiment"],
        )
        reviews = reviews.groupby(gby_desc)

        output = reviews.run(self.palimpzest_config())
        return output

    def _execute_q9(self) -> DataRecordCollection:
        """
        Execute Q9: Score from 1 to 5 how much did the reviewer like the movie based on the movie reviews for movie 'ant_man_and_the_wasp_quantumania'.

        Returns:
            DataFrame with columns: reviewId, reviewScore
        """
        # Load and filter reviews data for specific movie
        reviews_df = self.load_data("Reviews.csv").rename(
            columns={"id": "movieId"}
        )
        filtered_reviews = reviews_df[
            reviews_df["movieId"] == "ant_man_and_the_wasp_quantumania"
        ]

        # Create dataset
        reviews = pz.MemoryDataset(id="reviews", vals=filtered_reviews)

        # Add score column using detailed rubric from Lotus
        reviews = reviews.sem_add_columns(
            [
                {
                    "name": "reviewScore",
                    "type": int,
                    "desc": """Score from 1 to 5 how much did the reviewer like the movie based on provided rubrics.

Rubrics:
5: Very positive. Strong positive sentiment, indicating high satisfaction.
4: Positive. Noticeably positive sentiment, indicating general satisfaction.
3: Neutral. Expresses no clear positive or negative sentiment. May be factual or descriptive without emotional language.
2: Negative. Noticeably negative sentiment, indicating some level of dissatisfaction but without strong anger or frustration.
1: Very negative. Strong negative sentiment, indicating high dissatisfaction, frustration, or anger.

Review: {reviewText}

Only provide the score number (1-5) with no other comments.""",
                }
            ],
            depends_on=["reviewText"],
        )

        # Project to required columns
        reviews = reviews.project(["reviewId", "reviewScore"])

        output = reviews.run(self.palimpzest_config())
        return output

    def _execute_q10(self) -> DataRecordCollection:
        """
        Execute Q10: Rank the movies based on movie reviews. For each movie, score every review of it from 1 to 5, then calculate the average score of these reviews for each movie.

        Returns:
            DataFrame with columns: movieId, movieScore
        """
        # Load all reviews data
        reviews_df = self.load_data("Reviews.csv").rename(
            columns={"id": "movieId"}
        )

        # Create dataset
        reviews = pz.MemoryDataset(id="reviews", vals=reviews_df)

        # Add score column using detailed rubric from Lotus
        reviews = reviews.sem_add_columns(
            [
                {
                    "name": "reviewScore",
                    "type": int,
                    "desc": """Score from 1 to 5 how much did the reviewer like the movie based on provided rubrics.

Rubrics:
5: Very positive. Strong positive sentiment, indicating high satisfaction.
4: Positive. Noticeably positive sentiment, indicating general satisfaction.
3: Neutral. Expresses no clear positive or negative sentiment. May be factual or descriptive without emotional language.
2: Negative. Noticeably negative sentiment, indicating some level of dissatisfaction but without strong anger or frustration.
1: Very negative. Strong negative sentiment, indicating high dissatisfaction, frustration, or anger.

Review: {reviewText}

Only provide the score number (1-5) with no other comments.""",
                }
            ],
            depends_on=["reviewText"],
        )

        # Project to movieId and reviewScore, then group by movieId to get average
        reviews = reviews.project(["movieId", "reviewScore"])
        gby_desc = GroupBySig(
            group_by_fields=["movieId"],
            agg_funcs=["average"],
            agg_fields=["reviewScore"],
        )
        reviews = reviews.groupby(gby_desc)

        output = reviews.run(self.palimpzest_config())
        return output
