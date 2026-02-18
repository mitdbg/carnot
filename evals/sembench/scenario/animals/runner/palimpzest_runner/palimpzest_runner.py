"""
Created on Aug 8, 2025

@author: Jiale Lao

Palimpzest runner implementation for animals use case.
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

PANDAS_DTYPE_TO_PYDANTIC = {
    "object": str,
    "bool": bool,
    "int64": int,
    "float64": float,
}


class PalimpzestRunner(GenericPalimpzestRunner):
    """Palimpzest runner for animals use case."""

    def __init__(
        self,
        use_case: str,
        scale_factor: int,
        model_name: str = "gemini-2.5-flash",
        concurrent_llm_worker=20,
        skip_setup: bool = False,
    ):
        """
        Initialize Palimpzest runner for animals case.

        Args:
            use_case: The use case to run
            model_name: LLM model to use
            concurrent_llm_worker: Number of concurrent workers
        """
        super().__init__(
            use_case, scale_factor, model_name, concurrent_llm_worker, skip_setup
        )

    def _execute_q1(self) -> DataRecordCollection:
        """
        Q1: How many pictures of zebras do we have in our database?
        Ground truth: select count(*) from ImageData where Species LIKE '%ZEBRA%';

        Returns:
            DataFrame with columns: count(*)
        """
        # Load image data and rename columns to match schema
        image_data = self.load_data("image_data.csv").rename(
            columns={
                "ImagePath": "image",
                "City": "city",
                "StationID": "stationID",
            }
        )

        # Define schema with proper types
        image_cols = [
            {
                "name": col,
                "type": (
                    pz.ImageFilepath
                    if col == "image"
                    else PANDAS_DTYPE_TO_PYDANTIC[str(dtype)]
                ),
                "desc": f"The {col} column from image data",
            }
            for col, dtype in zip(image_data.columns, image_data.dtypes)
        ]

        images = pz.MemoryDataset(
            id="images", vals=image_data, schema=image_cols
        )
        images = images.sem_filter(
            "Determine if this image contains a zebra", depends_on=["image"]
        )
        images = images.count()

        output = images.run(self.palimpzest_config())

        return output

    def _execute_q2(self) -> DataRecordCollection:
        """
        Q2: How many sound recordings of elephants do we have in our database?
        Ground truth: select count(*) from AudioData where Animal = 'Elephant';

        Returns:
            DataFrame with columns: count(*)
        """
        # Load audio data
        audio_data = self.load_data("audio_data.csv").rename(
            columns={
                "AudioPath": "audio",
                "City": "city",
                "StationID": "stationID",
            }
        )

        # Define schema with proper types
        audio_cols = [
            {
                "name": col,
                "type": (
                    pz.AudioFilepath
                    if col == "audio"
                    else PANDAS_DTYPE_TO_PYDANTIC[str(dtype)]
                ),
                "desc": f"The {col} column from audio data",
            }
            for col, dtype in zip(audio_data.columns, audio_data.dtypes)
        ]

        audios = pz.MemoryDataset(
            id="audios", vals=audio_data, schema=audio_cols
        )
        audios = audios.sem_filter(
            "Determine if this audio recording contains elephant sounds",
            depends_on=["audio"],
        )
        audios = audios.count()

        output = audios.run(self.palimpzest_config())

        return output

    def _execute_q3(self) -> DataRecordCollection:
        """
        Q3: What is the city where we captured most pictures of zebras?
        Ground truth: select city from ImageData where Species LIKE '%ZEBRA%' group by city order by count(*) desc limit 1;

        Returns:
            DataFrame with columns: city
        """
        image_data = self.load_data("image_data.csv").rename(
            columns={
                "ImagePath": "image",
                "City": "city",
                "StationID": "stationID",
            }
        )

        # Define schema with proper types
        image_cols = [
            {
                "name": col,
                "type": (
                    pz.ImageFilepath
                    if col == "image"
                    else PANDAS_DTYPE_TO_PYDANTIC[str(dtype)]
                ),
                "desc": f"The {col} column from image data",
            }
            for col, dtype in zip(image_data.columns, image_data.dtypes)
        ]

        images = pz.MemoryDataset(
            id="images", vals=image_data, schema=image_cols
        )
        images = images.sem_filter(
            "Determine if this image contains a zebra", depends_on=["image"]
        )
        images = images.project(["city"])
        gby_desc = GroupBySig(
            group_by_fields=["city"], agg_funcs=["count"], agg_fields=["city"]
        )
        images = images.groupby(gby_desc)

        # Execute the groupby to get results
        grouped_result = images.run(self.palimpzest_config())

        # Preserve execution statistics before processing
        exec_stats = grouped_result.execution_stats

        # Convert to DataFrame for post-processing
        grouped_df = grouped_result.to_df()

        if not grouped_df.empty:
            # Find the maximum count
            max_count = grouped_df["count(city)"].max()
            # Get all cities with the maximum count (handles ties)
            max_cities_df = grouped_df[grouped_df["count(city)"] == max_count][
                ["city"]
            ]
        else:
            max_cities_df = pd.DataFrame(columns=["city"])

        # Create new result with processed data but preserve execution stats
        result_cols = [
            {
                "name": "city",
                "type": str,
                "desc": "The city column for cities with maximum zebra count",
            }
        ]

        final_dataset = pz.MemoryDataset(
            id="max_cities", vals=max_cities_df, schema=result_cols
        )
        final_result = final_dataset.run(self.palimpzest_config())

        # Restore the original execution statistics
        final_result.execution_stats = exec_stats

        return final_result

    def _execute_q4(self) -> DataRecordCollection:
        """
        Q4: What is the city for which we have most recordings of elephants?
        Ground truth: select city from AudioData where Animal = 'Elephant' group by city order by count(*) desc limit 1;

        Returns:
            DataFrame with columns: city
        """
        audio_data = self.load_data("audio_data.csv").rename(
            columns={
                "AudioPath": "audio",
                "City": "city",
                "StationID": "stationID",
            }
        )

        # Define schema with proper types
        audio_cols = [
            {
                "name": col,
                "type": (
                    pz.AudioFilepath
                    if col == "audio"
                    else PANDAS_DTYPE_TO_PYDANTIC[str(dtype)]
                ),
                "desc": f"The {col} column from audio data",
            }
            for col, dtype in zip(audio_data.columns, audio_data.dtypes)
        ]

        audios = pz.MemoryDataset(
            id="audios", vals=audio_data, schema=audio_cols
        )
        audios = audios.sem_filter(
            "Determine if this audio recording contains elephant sounds",
            depends_on=["audio"],
        )
        audios = audios.project(["city"])
        gby_desc = GroupBySig(
            group_by_fields=["city"], agg_funcs=["count"], agg_fields=["city"]
        )
        audios = audios.groupby(gby_desc)

        # Execute the groupby to get results
        grouped_result = audios.run(self.palimpzest_config())

        # Preserve execution statistics before processing
        exec_stats = grouped_result.execution_stats

        # Convert to DataFrame for post-processing
        grouped_df = grouped_result.to_df()

        if not grouped_df.empty:
            # Find the maximum count
            max_count = grouped_df["count(city)"].max()
            # Get all cities with the maximum count (handles ties)
            max_cities_df = grouped_df[grouped_df["count(city)"] == max_count][
                ["city"]
            ]
        else:
            max_cities_df = pd.DataFrame(columns=["city"])

        # Create new result with processed data but preserve execution stats
        result_cols = [
            {
                "name": "city",
                "type": str,
                "desc": "The city column for cities with maximum elephant audio count",
            }
        ]

        final_dataset = pz.MemoryDataset(
            id="max_cities", vals=max_cities_df, schema=result_cols
        )
        final_result = final_dataset.run(self.palimpzest_config())

        # Restore the original execution statistics
        final_result.execution_stats = exec_stats

        return final_result

    def _execute_q5(self) -> DataRecordCollection:
        """
        Q5: What is the list of cities for which we have either images or audio recordings of elephants?
        Ground truth: select distinct city from (select city from ImageData where Species LIKE '%ELEPHANT%') UNION (select city from AudioData where Animal = 'Elephant');

        Returns:
            DataFrame with columns: city
        """
        image_data = self.load_data("image_data.csv").rename(
            columns={
                "ImagePath": "image",
                "City": "city",
                "StationID": "stationID",
            }
        )

        audio_data = self.load_data("audio_data.csv").rename(
            columns={
                "AudioPath": "audio",
                "City": "city",
                "StationID": "stationID",
            }
        )

        # Define schemas
        image_cols = [
            {
                "name": col,
                "type": (
                    pz.ImageFilepath
                    if col == "image"
                    else PANDAS_DTYPE_TO_PYDANTIC[str(dtype)]
                ),
                "desc": f"The {col} column from image data",
            }
            for col, dtype in zip(image_data.columns, image_data.dtypes)
        ]

        audio_cols = [
            {
                "name": col,
                "type": (
                    pz.AudioFilepath
                    if col == "audio"
                    else PANDAS_DTYPE_TO_PYDANTIC[str(dtype)]
                ),
                "desc": f"The {col} column from audio data",
            }
            for col, dtype in zip(audio_data.columns, audio_data.dtypes)
        ]

        # Get cities with elephant images
        images = pz.MemoryDataset(
            id="images", vals=image_data, schema=image_cols
        )
        elephant_images = images.sem_filter(
            "Determine if this image contains an elephant", depends_on=["image"]
        )
        elephant_image_cities = elephant_images.project(["city"])

        # Get cities with elephant audio
        audios = pz.MemoryDataset(
            id="audios", vals=audio_data, schema=audio_cols
        )
        elephant_audios = audios.sem_filter(
            "Determine if this audio recording contains elephant sounds",
            depends_on=["audio"],
        )
        elephant_audio_cities = elephant_audios.project(["city"])

        # Execute queries and preserve execution statistics
        image_result = elephant_image_cities.run(self.palimpzest_config())
        audio_result = elephant_audio_cities.run(self.palimpzest_config())

        # Preserve execution statistics (combine both)
        exec_stats = image_result.execution_stats
        if (
            hasattr(audio_result, "execution_stats")
            and audio_result.execution_stats
        ):
            # Combine execution statistics from both queries
            if hasattr(exec_stats, "update"):
                exec_stats.update(audio_result.execution_stats)

        # Convert to DataFrames and get distinct values
        image_cities_df = image_result.to_df().drop_duplicates()
        audio_cities_df = audio_result.to_df().drop_duplicates()

        # Combine and get unique cities
        combined_cities = pd.concat(
            [image_cities_df, audio_cities_df]
        ).drop_duplicates()

        # Create result schema
        result_cols = [
            {
                "name": "city",
                "type": str,
                "desc": "The city column for cities with elephant images or audio",
            }
        ]

        final_dataset = pz.MemoryDataset(
            id="combined_cities", vals=combined_cities, schema=result_cols
        )
        final_result = final_dataset.run(self.palimpzest_config())

        # Restore the original execution statistics
        final_result.execution_stats = exec_stats

        return final_result

    def _execute_q6(self) -> DataRecordCollection:
        """
        Q6: For which cities do we have images of monkeys but no audio recordings of monkeys?
        Ground truth: select distinct city from ImageData I where Species LIKE '%MONKEY%' and not exists (select * from AudioData A where A.city = I.city and A.animal = 'Monkey');

        Returns:
            DataFrame with columns: city
        """
        image_data = self.load_data("image_data.csv").rename(
            columns={
                "ImagePath": "image",
                "City": "city",
                "StationID": "stationID",
            }
        )

        audio_data = self.load_data("audio_data.csv").rename(
            columns={
                "AudioPath": "audio",
                "City": "city",
                "StationID": "stationID",
            }
        )

        # Define schemas
        image_cols = [
            {
                "name": col,
                "type": (
                    pz.ImageFilepath
                    if col == "image"
                    else PANDAS_DTYPE_TO_PYDANTIC[str(dtype)]
                ),
                "desc": f"The {col} column from image data",
            }
            for col, dtype in zip(image_data.columns, image_data.dtypes)
        ]

        audio_cols = [
            {
                "name": col,
                "type": (
                    pz.AudioFilepath
                    if col == "audio"
                    else PANDAS_DTYPE_TO_PYDANTIC[str(dtype)]
                ),
                "desc": f"The {col} column from audio data",
            }
            for col, dtype in zip(audio_data.columns, audio_data.dtypes)
        ]

        # Get cities with monkey images
        images = pz.MemoryDataset(
            id="images", vals=image_data, schema=image_cols
        )
        monkey_images = images.sem_filter(
            "Determine if this image contains a monkey", depends_on=["image"]
        )
        monkey_image_cities = monkey_images.project(["city"])

        # Get cities with monkey audio
        audios = pz.MemoryDataset(
            id="audios", vals=audio_data, schema=audio_cols
        )
        monkey_audios = audios.sem_filter(
            "Determine if this audio recording contains monkey sounds",
            depends_on=["audio"],
        )
        monkey_audio_cities = monkey_audios.project(["city"])

        # Execute queries and preserve execution statistics
        image_result = monkey_image_cities.run(self.palimpzest_config())
        audio_result = monkey_audio_cities.run(self.palimpzest_config())

        # Preserve execution statistics (combine both)
        exec_stats = image_result.execution_stats
        if (
            hasattr(audio_result, "execution_stats")
            and audio_result.execution_stats
        ):
            if hasattr(exec_stats, "update"):
                exec_stats.update(audio_result.execution_stats)

        # Get the results as DataFrames with distinct values
        image_cities_df = image_result.to_df().drop_duplicates()
        audio_cities_df = audio_result.to_df().drop_duplicates()

        # Cities with monkey images but not audio
        if not audio_cities_df.empty:
            result_cities = image_cities_df[
                ~image_cities_df["city"].isin(audio_cities_df["city"])
            ]
        else:
            result_cities = image_cities_df

        # Create result schema
        result_cols = [
            {
                "name": "city",
                "type": str,
                "desc": "The city column for cities with monkey images but no audio",
            }
        ]

        final_dataset = pz.MemoryDataset(
            id="result_cities", vals=result_cities, schema=result_cols
        )
        final_result = final_dataset.run(self.palimpzest_config())

        # Restore the original execution statistics
        final_result.execution_stats = exec_stats

        return final_result

    def _execute_q7(self) -> DataRecordCollection:
        """
        Q7: What are the cities for which zebras and impala co-occur?
        Ground truth: select city from (select city from ImageData where Species LIKE '%ZEBRA%') INTERSECT (select city from ImageData where Species LIKE '%IMPALA%');

        Returns:
            DataFrame with columns: city
        """
        image_data = self.load_data("image_data.csv").rename(
            columns={
                "ImagePath": "image",
                "City": "city",
                "StationID": "stationID",
            }
        )

        # Define schema
        image_cols = [
            {
                "name": col,
                "type": (
                    pz.ImageFilepath
                    if col == "image"
                    else PANDAS_DTYPE_TO_PYDANTIC[str(dtype)]
                ),
                "desc": f"The {col} column from image data",
            }
            for col, dtype in zip(image_data.columns, image_data.dtypes)
        ]

        # Get cities with zebras
        zebra_images = pz.MemoryDataset(
            id="zebra_images", vals=image_data, schema=image_cols
        )
        zebra_images = zebra_images.sem_filter(
            "Determine if this image contains a zebra", depends_on=["image"]
        )
        zebra_cities = zebra_images.project(["city"])

        # Get cities with impalas
        impala_images = pz.MemoryDataset(
            id="impala_images", vals=image_data, schema=image_cols
        )
        impala_images = impala_images.sem_filter(
            "Determine if this image contains an impala", depends_on=["image"]
        )
        impala_cities = impala_images.project(["city"])

        # Execute queries and preserve execution statistics
        zebra_result = zebra_cities.run(self.palimpzest_config())
        impala_result = impala_cities.run(self.palimpzest_config())

        # Preserve execution statistics (combine both)
        exec_stats = zebra_result.execution_stats
        if (
            hasattr(impala_result, "execution_stats")
            and impala_result.execution_stats
        ):
            if hasattr(exec_stats, "update"):
                exec_stats.update(impala_result.execution_stats)

        # Find intersection - cities that have both zebras and impalas
        zebra_df = zebra_result.to_df().drop_duplicates()
        impala_df = impala_result.to_df().drop_duplicates()

        # Find common cities
        common_cities = pd.merge(zebra_df, impala_df, on="city", how="inner")

        # Create result schema
        result_cols = [
            {
                "name": "city",
                "type": str,
                "desc": "The city column for cities with both zebras and impalas",
            }
        ]

        final_dataset = pz.MemoryDataset(
            id="common_cities", vals=common_cities, schema=result_cols
        )
        final_result = final_dataset.run(self.palimpzest_config())

        # Restore the original execution statistics
        final_result.execution_stats = exec_stats

        return final_result

    def _execute_q8(self) -> DataRecordCollection:
        """
        Q8: What is the list of cities for which we found at least one image or audio recording of elephants and at least one image or audio recording of monkeys?
        Ground truth: select city from ((select city from ImageData where Species LIKE '%ELEPHANT%') UNION (select city from AudioData where Animal = 'Elephant')) INTERSECT ((select city from ImageData where Species LIKE '%MONKEY%') UNION (select city from AudioData where Animal = 'Monkey'));

        Returns:
            DataFrame with columns: city
        """
        image_data = self.load_data("image_data.csv").rename(
            columns={
                "ImagePath": "image",
                "City": "city",
                "StationID": "stationID",
            }
        )

        audio_data = self.load_data("audio_data.csv").rename(
            columns={
                "AudioPath": "audio",
                "City": "city",
                "StationID": "stationID",
            }
        )

        # Define schemas
        image_cols = [
            {
                "name": col,
                "type": (
                    pz.ImageFilepath
                    if col == "image"
                    else PANDAS_DTYPE_TO_PYDANTIC[str(dtype)]
                ),
                "desc": f"The {col} column from image data",
            }
            for col, dtype in zip(image_data.columns, image_data.dtypes)
        ]

        audio_cols = [
            {
                "name": col,
                "type": (
                    pz.AudioFilepath
                    if col == "audio"
                    else PANDAS_DTYPE_TO_PYDANTIC[str(dtype)]
                ),
                "desc": f"The {col} column from audio data",
            }
            for col, dtype in zip(audio_data.columns, audio_data.dtypes)
        ]

        # Get cities with elephants (images or audio)
        images = pz.MemoryDataset(
            id="images", vals=image_data, schema=image_cols
        )
        elephant_images = images.sem_filter(
            "Determine if this image contains an elephant", depends_on=["image"]
        )
        elephant_image_cities = elephant_images.project(["city"])

        audios = pz.MemoryDataset(
            id="audios", vals=audio_data, schema=audio_cols
        )
        elephant_audios = audios.sem_filter(
            "Determine if this audio recording contains elephant sounds",
            depends_on=["audio"],
        )
        elephant_audio_cities = elephant_audios.project(["city"])

        # Get cities with monkeys (images or audio)
        monkey_images = images.sem_filter(
            "Determine if this image contains a monkey", depends_on=["image"]
        )
        monkey_image_cities = monkey_images.project(["city"])

        monkey_audios = audios.sem_filter(
            "Determine if this audio recording contains monkey sounds",
            depends_on=["audio"],
        )
        monkey_audio_cities = monkey_audios.project(["city"])

        # Execute queries and preserve execution statistics
        elephant_img_result = elephant_image_cities.run(
            self.palimpzest_config()
        )
        elephant_aud_result = elephant_audio_cities.run(
            self.palimpzest_config()
        )
        monkey_img_result = monkey_image_cities.run(self.palimpzest_config())
        monkey_aud_result = monkey_audio_cities.run(self.palimpzest_config())

        # Preserve execution statistics (combine all)
        exec_stats = elephant_img_result.execution_stats
        for result in [
            elephant_aud_result,
            monkey_img_result,
            monkey_aud_result,
        ]:
            if hasattr(result, "execution_stats") and result.execution_stats:
                if hasattr(exec_stats, "update"):
                    exec_stats.update(result.execution_stats)

        # Get results as DataFrames with distinct values
        elephant_img_df = elephant_img_result.to_df().drop_duplicates()
        elephant_aud_df = elephant_aud_result.to_df().drop_duplicates()
        monkey_img_df = monkey_img_result.to_df().drop_duplicates()
        monkey_aud_df = monkey_aud_result.to_df().drop_duplicates()

        # Union elephant cities (images + audio)
        elephant_cities = pd.concat(
            [elephant_img_df, elephant_aud_df]
        ).drop_duplicates()

        # Union monkey cities (images + audio)
        monkey_cities = pd.concat(
            [monkey_img_df, monkey_aud_df]
        ).drop_duplicates()

        # Intersect the two sets
        result_cities = pd.merge(
            elephant_cities, monkey_cities, on="city", how="inner"
        )

        # Create result schema
        result_cols = [
            {
                "name": "city",
                "type": str,
                "desc": "The city column for cities with both elephants and monkeys",
            }
        ]

        final_dataset = pz.MemoryDataset(
            id="result_cities", vals=result_cities, schema=result_cols
        )
        final_result = final_dataset.run(self.palimpzest_config())

        # Restore the original execution statistics
        final_result.execution_stats = exec_stats

        return final_result

    def _execute_q9(self) -> DataRecordCollection:
        """
        Q9: What is the list of cities for which we have both images and audio recordings of monkeys?
        Ground truth: select distinct city from (select city from ImageData where Species LIKE '%MONKEY%') INTERSECT (select city from AudioData where Animal = 'Monkey');

        Returns:
            DataFrame with columns: city
        """
        image_data = self.load_data("image_data.csv").rename(
            columns={
                "ImagePath": "image",
                "City": "city",
                "StationID": "stationID",
            }
        )

        audio_data = self.load_data("audio_data.csv").rename(
            columns={
                "AudioPath": "audio",
                "City": "city",
                "StationID": "stationID",
            }
        )

        # Define schemas
        image_cols = [
            {
                "name": col,
                "type": (
                    pz.ImageFilepath
                    if col == "image"
                    else PANDAS_DTYPE_TO_PYDANTIC[str(dtype)]
                ),
                "desc": f"The {col} column from image data",
            }
            for col, dtype in zip(image_data.columns, image_data.dtypes)
        ]

        audio_cols = [
            {
                "name": col,
                "type": (
                    pz.AudioFilepath
                    if col == "audio"
                    else PANDAS_DTYPE_TO_PYDANTIC[str(dtype)]
                ),
                "desc": f"The {col} column from audio data",
            }
            for col, dtype in zip(audio_data.columns, audio_data.dtypes)
        ]

        # Get cities with monkey images
        images = pz.MemoryDataset(
            id="images", vals=image_data, schema=image_cols
        )
        monkey_images = images.sem_filter(
            "Determine if this image contains a monkey", depends_on=["image"]
        )
        monkey_image_cities = monkey_images.project(["city"])

        # Get cities with monkey audio
        audios = pz.MemoryDataset(
            id="audios", vals=audio_data, schema=audio_cols
        )
        monkey_audios = audios.sem_filter(
            "Determine if this audio recording contains monkey sounds",
            depends_on=["audio"],
        )
        monkey_audio_cities = monkey_audios.project(["city"])

        # Execute queries and preserve execution statistics
        image_result = monkey_image_cities.run(self.palimpzest_config())
        audio_result = monkey_audio_cities.run(self.palimpzest_config())

        # Preserve execution statistics (combine both)
        exec_stats = image_result.execution_stats
        if (
            hasattr(audio_result, "execution_stats")
            and audio_result.execution_stats
        ):
            if hasattr(exec_stats, "update"):
                exec_stats.update(audio_result.execution_stats)

        # Get results as DataFrames with distinct values
        image_cities_df = image_result.to_df().drop_duplicates()
        audio_cities_df = audio_result.to_df().drop_duplicates()

        # Intersect the two sets
        result_cities = pd.merge(
            image_cities_df, audio_cities_df, on="city", how="inner"
        )

        # Create result schema
        result_cols = [
            {
                "name": "city",
                "type": str,
                "desc": "The city column for cities with both monkey images and audio",
            }
        ]

        final_dataset = pz.MemoryDataset(
            id="result_cities", vals=result_cities, schema=result_cols
        )
        final_result = final_dataset.run(self.palimpzest_config())

        # Restore the original execution statistics
        final_result.execution_stats = exec_stats

        return final_result

    def _execute_q10(self) -> DataRecordCollection:
        """
        Q10: What is the city and station with most associated pictures showing zebras?
        Ground truth: select city, stationID from ImageData where Species LIKE '%ZEBRA%' group by city, stationID order by count(*) DESC limit 1;

        Returns:
            DataFrame with columns: city, stationID
        """
        image_data = self.load_data("image_data.csv").rename(
            columns={
                "ImagePath": "image",
                "City": "city",
                "StationID": "stationID",
            }
        )

        # Define schema
        image_cols = [
            {
                "name": col,
                "type": (
                    pz.ImageFilepath
                    if col == "image"
                    else PANDAS_DTYPE_TO_PYDANTIC[str(dtype)]
                ),
                "desc": f"The {col} column from image data",
            }
            for col, dtype in zip(image_data.columns, image_data.dtypes)
        ]

        images = pz.MemoryDataset(
            id="images", vals=image_data, schema=image_cols
        )
        images = images.sem_filter(
            "Determine if this image contains a zebra", depends_on=["image"]
        )
        images = images.project(["city", "stationID"])
        gby_desc = GroupBySig(
            group_by_fields=["city", "stationID"],
            agg_funcs=["count"],
            agg_fields=["city"],
        )
        images = images.groupby(gby_desc)

        # Execute the groupby to get results
        grouped_result = images.run(self.palimpzest_config())

        # Preserve execution statistics before processing
        exec_stats = grouped_result.execution_stats

        # Convert to DataFrame for post-processing
        grouped_df = grouped_result.to_df()

        if not grouped_df.empty:
            # Find the maximum count
            max_count = grouped_df["count(city)"].max()
            # Get all (city, station) pairs with the maximum count (handles ties)
            max_pairs_df = grouped_df[grouped_df["count(city)"] == max_count][
                ["city", "stationID"]
            ]
        else:
            max_pairs_df = pd.DataFrame(columns=["city", "stationID"])

        # Create new result with processed data but preserve execution stats
        result_cols = [
            {
                "name": "city",
                "type": str,
                "desc": "The city column for (city, station) pairs with maximum zebra count",
            },
            {
                "name": "stationID",
                "type": str,
                "desc": "The stationID column for (city, station) pairs with maximum zebra count",
            },
        ]

        final_dataset = pz.MemoryDataset(
            id="max_pairs", vals=max_pairs_df, schema=result_cols
        )
        final_result = final_dataset.run(self.palimpzest_config())

        # Restore the original execution statistics
        final_result.execution_stats = exec_stats

        return final_result

    def _get_empty_results_dataframe(self, query_id: int) -> pd.DataFrame:
        """
        Get empty DataFrame with correct columns for a query.

        Args:
            query_id: ID of the query

        Returns:
            Empty DataFrame with correct columns
        """
        if query_id == 1:
            return pd.DataFrame(columns=["count(*)"])
        elif query_id == 2:
            return pd.DataFrame(columns=["count(*)"])
        elif query_id == 3:
            return pd.DataFrame(columns=["city"])
        elif query_id == 4:
            return pd.DataFrame(columns=["city"])
        elif query_id == 5:
            return pd.DataFrame(columns=["city"])
        elif query_id == 6:
            return pd.DataFrame(columns=["city"])
        elif query_id == 7:
            return pd.DataFrame(columns=["city"])
        elif query_id == 8:
            return pd.DataFrame(columns=["city"])
        elif query_id == 9:
            return pd.DataFrame(columns=["city"])
        elif query_id == 10:
            return pd.DataFrame(columns=["city", "stationID"])
        else:
            return pd.DataFrame()
