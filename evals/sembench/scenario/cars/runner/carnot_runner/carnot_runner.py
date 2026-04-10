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
    def __init__(
        self,
        use_case: str,
        scale_factor: int,
        model_name: str = "gpt-4o-audio-preview",
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
        """Q1: Find cars that were in a crash/accident/collision."""
        # Load data
        complaints_df = self.load_data(f"text_complaints_data_{self.scale_factor}.csv")
        dataset = carnot.Dataset(
            name="Complaints Data",
            annotation="",
            items=complaints_df.to_dict(orient="records"),
        )
        execution = carnot.Execution(
            query="Find cars that were in a crash/accident/collision. Return the (distinct) car_id of each such car.",
            datasets=[dataset],
            llm_config=self.llm_config,
        )
        _, plan = execution.plan()
        execution._plan = plan
        output, _ = execution.run()

        return output, {"total_tokens": 0, "total_execution_cost": 0.0}

    def _execute_q2(self) -> tuple[list[dict], dict]:
        """Q2: Find electric cars with available audio recordings that show a dead battery."""
        # TODO: come back once carnot supports audio data
        return [], {"total_tokens": 0, "total_execution_cost": 0.0}

        # # define the schema for the data you read from the file
        # data_cols = [
        #     {"name": "audio_path", "type": AudioFilepath, "desc": "The filepath containing audios."},
        #     {"name": "car_id", "type": int, "desc": "The integer id for the car."},
        #     {"name": "audio_id", "type": int, "desc": "The integer id for the audio."},
        #     {"name": "fuel_type", "type": str, "desc": "The string fuel_type."},
        # ]

        # class MyDataset(pz.IterDataset):
        # def __init__(self, id: str, car_df: pd.DataFrame):
        #     super().__init__(id=id, schema=data_cols)
        #     self.car_df = car_df

        # def __len__(self):
        #     return len(self.car_df)

        # def __getitem__(self, idx: int):
        #     # get row from dataframe
        #     return self.car_df.iloc[idx].to_dict()


        # def run(pz_config, data_dir: str, scale_factor: int = 157376):
        #     # Load data
        #     audio = pd.read_csv(os.path.join(data_dir, f"data/sf_{scale_factor}/audio_car_data_{scale_factor}.csv"))
        #     cars = pd.read_csv(os.path.join(data_dir, f"data/sf_{scale_factor}/car_data_{scale_factor}.csv")) 
            
        #     # Join
        #     tmp_join = cars.join(audio.set_index('car_id'), on='car_id', how='inner')[['audio_path', 'car_id', 'audio_id', 'fuel_type']]

        #     tmp_join = MyDataset(id="my-data", car_df=tmp_join)

        #     # Filter fuel_type
        #     tmp_join = tmp_join.filter(lambda row: row['fuel_type'] == 'Electric')

        #     # Filter audio
        #     tmp_join = tmp_join.sem_filter('You are given an audio recording of car diagnostics. Return true if the car from the recording has a dead battery, false otherwise.', depends_on=['audio_path'])
        #     tmp_join = tmp_join.project(['car_id'])
        #     tmp_join = tmp_join.distinct(distinct_cols=['car_id'])
            
        #     output = tmp_join.run(pz_config)
        #     return output

    def _execute_q3(self) -> tuple[list[dict], dict]:
        """Q3: Find ten cars with manual transmission that are not damaged according to images. Return the VIN."""
        # TODO: come back once carnot supports image data
        return [], {"total_tokens": 0, "total_execution_cost": 0.0}

        # # define the schema for the data you read from the file
        # car_image_data_cols = [
        #     {"name": "image_path", "type": ImageFilepath, "desc": "The filepath containing the car image"},
        #     {"name": "car_id", "type": int, "desc": "The integer id for the car"},
        #     {"name": "image_id", "type": int, "desc": "The integer id for the car image"},
        #     {"name": "transmission", "type": str, "desc": "The string transmission type"},
        #     {"name": "vin", "type": str, "desc": "The vehicle identification number"},
        # ]


        # class MyDataset(pz.IterDataset):
        # def __init__(self, id: str, car_df: pd.DataFrame):
        #     super().__init__(id=id, schema=car_image_data_cols)
        #     self.car_df = car_df

        # def __len__(self):
        #     return len(self.car_df)

        # def __getitem__(self, idx: int):
        #     # get row from dataframe
        #     return self.car_df.iloc[idx].to_dict()
        

        # def run(pz_config, data_dir: str, scale_factor: int = 157376):
        #     # Load data
        #     car_images = pd.read_csv(os.path.join(data_dir, f"data/sf_{scale_factor}/image_car_data_{scale_factor}.csv")) 
        #     cars = pd.read_csv(os.path.join(data_dir, f"data/sf_{scale_factor}/car_data_{scale_factor}.csv")) 
            
        #     # Join before since PZ does not support joins
        #     tmp_join = cars.join(car_images.set_index('car_id'), on='car_id', how='inner')[['image_path', 'car_id', 'image_id', 'transmission', 'vin']]

        #     tmp_join = MyDataset(id="my-car-data", car_df=tmp_join)

        #     # Filter transmission
        #     tmp_join = tmp_join.filter(lambda row: row['transmission'] == 'Manual')

        #     # Filter image
        #     tmp_join = tmp_join.sem_filter('You are given an image of a vehicle or its parts. Return true if car is not damaged.', depends_on=['image_path'])
        #     tmp_join = tmp_join.project(['vin'])
        #     candidates = tmp_join.limit(10)

        #     output = candidates.run(pz_config)
        #     return output

    def _execute_q4(self) -> tuple[list[dict], dict]:
        """Q4: What is the average age of cars with engine problems?"""
        # Load data
        complaints_df = self.load_data(f"text_complaints_data_{self.scale_factor}.csv")
        cars_df = self.load_data(f"car_data_{self.scale_factor}.csv")
        complaints_dataset = carnot.Dataset(
            name="Complaints Data",
            annotation="",
            items=complaints_df.to_dict(orient="records"),
        )
        cars_dataset = carnot.Dataset(
            name="Cars Data",
            annotation="",
            items=cars_df.to_dict(orient="records"),
        )
        execution = carnot.Execution(
            query="What is the average age in years of cars with engine problems? Return a the value in a column named average_age. Assume the current year is 2026.",
            datasets=[complaints_dataset, cars_dataset],
            llm_config=self.llm_config,
        )
        _, plan = execution.plan()
        execution._plan = plan
        output, _ = execution.run()

        return output, {"total_tokens": 0, "total_execution_cost": 0.0}

    def _execute_q5(self) -> tuple[list[dict], dict]:
        """Q5: How many automatic cars are damaged according to both audio and images?"""
        # TODO: come back once carnot supports image and audio data
        return [], {"total_tokens": 0, "total_execution_cost": 0.0}

        # data_cols = [
        #     {"name": "car_id", "type": int, "desc": "The integer id for the car"},
        #     {"name": "transmission", "type": str, "desc": "The string transmission."},
        #     {"name": "image_id", "type": int, "desc": "The integer id for the car image"},
        #     {"name": "image_path", "type": ImageFilepath, "desc": "The filepath containing the car image"},
        #     {"name": "audio_id", "type": int, "desc": "The integer id for the audio"},
        #     {"name": "audio_path", "type": AudioFilepath, "desc": "The filepath containing audios."},
        # ]


        # class MyDataset(pz.IterDataset):
        # def __init__(self, id: str, car_df: pd.DataFrame):
        #     super().__init__(id=id, schema=data_cols)
        #     self.car_df = car_df

        # def __len__(self):
        #     return len(self.car_df)

        # def __getitem__(self, idx: int):
        #     # get row from dataframe
        #     return self.car_df.iloc[idx].to_dict()
        

        # def run(pz_config, data_dir: str, scale_factor: int = 157376):
        #     # Load data
        #     cars = pd.read_csv(os.path.join(data_dir, f"data/sf_{scale_factor}/car_data_{scale_factor}.csv")) 
        #     car_images = pd.read_csv(os.path.join(data_dir, f"data/sf_{scale_factor}/image_car_data_{scale_factor}.csv"))
        #     audio = pd.read_csv(os.path.join(data_dir, f"data/sf_{scale_factor}/audio_car_data_{scale_factor}.csv"))
            
        #     # Join 
        #     tmp_join = cars.join(car_images.set_index('car_id'), on='car_id', how='inner')[['image_path', 'car_id', 'image_id', 'transmission']]
        #     tmp_join = tmp_join.join(audio.set_index('car_id'), on='car_id', how='inner')[['car_id', 'transmission', 'image_id', 'image_path', 'audio_path', 'audio_id']]
            
        #     tmp_join = MyDataset(id="my-data", car_df=tmp_join)

        #     # Filter transmission
        #     tmp_join = tmp_join.filter(lambda row: row['transmission'] == 'Automatic')

        #     # Filter audio
        #     tmp_join = tmp_join.sem_filter('You are given an audio recording of car diagnostics. Return true if the recording captures an audio of a damaged car.', depends_on=['audio_path'])
        #     # Filter image
        #     tmp_join = tmp_join.sem_filter("You are given an image of a vehicle or its parts. Return true if car is damaged.", depends_on=['image_path'])
        #     tmp_join = tmp_join.project(['car_id', 'transmission'])
        #     tmp_join = tmp_join.distinct(distinct_cols=['car_id', 'transmission'])

        #     # Group by
        #     gby_desc = GroupBySig(group_by_fields=["transmission"], agg_funcs=["count"], agg_fields=["transmission"])
        #     res = tmp_join.groupby(gby_desc)

        #     output = res.run(pz_config)
        #     return output

    def _execute_q6(self) -> tuple[list[dict], dict]:
        """Q6: Find cars that are damaged according to one modality but not the other. For this query, for complaints, check if the car was on fire."""
        # TODO: come back once carnot supports image and audio data
        return [], {"total_tokens": 0, "total_execution_cost": 0.0}
    
        # data_audio_cols = [
        #     {"name": "car_id", "type": int, "desc": "The integer id for the car"},
        #     {"name": "audio_path", "type": AudioFilepath, "desc": "The filepath containing audios."},
        # ]

        # data_image_cols = [
        #     {"name": "image_id", "type": int, "desc": "The integer id for the car image"},
        #     {"name": "car_id", "type": int, "desc": "The integer id for the car"},
        #     {"name": "image_path", "type": ImageFilepath, "desc": "The filepath containing the car image."},
        # ]


        # class MyDataset(pz.IterDataset):
        # def __init__(self, id: str, car_df: pd.DataFrame, schema: list):
        #     super().__init__(id=id, schema=schema)
        #     self.car_df = car_df

        # def __len__(self):
        #     return len(self.car_df)

        # def __getitem__(self, idx: int):
        #     # get row from dataframe
        #     return self.car_df.iloc[idx].to_dict()
        

        # def run(pz_config, data_dir: str, scale_factor: int = 157376):
        #     # Load data
        #     cars = pd.read_csv(os.path.join(data_dir, f"data/sf_{scale_factor}/car_data_{scale_factor}.csv")) 
        #     car_images = pd.read_csv(os.path.join(data_dir, f"data/sf_{scale_factor}/image_car_data_{scale_factor}.csv"))
        #     audio = pd.read_csv(os.path.join(data_dir, f"data/sf_{scale_factor}/audio_car_data_{scale_factor}.csv"))
        #     complaints = pd.read_csv(os.path.join(data_dir, f"data/sf_{scale_factor}/text_complaints_data_{scale_factor}.csv"))

        #     # Join with outer joins to get all combinations
        #     tmp_join = cars.join(car_images.set_index('car_id'), on='car_id', how='outer')[['image_path', 'car_id', 'image_id', 'year']]
        #     tmp_join = tmp_join.join(audio.set_index('car_id'), on='car_id', how='outer')[['car_id', 'year', 'image_id', 'image_path', 'audio_path', 'audio_id']]
        #     tmp_join = tmp_join.join(complaints.set_index('car_id'), on='car_id', how='outer')[['car_id', 'year', 'image_id', 'image_path', 'audio_path', 'audio_id', 'complaint_id', 'summary']]

        #     tmp_join["non_missing_count"] = tmp_join[["image_id", "complaint_id", "audio_id"]].notna().sum(axis=1)

        #     # Filter: keep rows with at least 2 present
        #     tmp_join = tmp_join[tmp_join["non_missing_count"] >= 2]
        #     tmp_join = tmp_join.drop(columns=["non_missing_count"])

        #     # Filter each modality separately
        #     audio_filtered = audio.loc[audio["car_id"].isin(tmp_join["car_id"].tolist()), ["car_id", "audio_path"]]
        #     car_images_filtered = car_images[car_images["car_id"].isin(tmp_join["car_id"].tolist())]
        #     complaints_filtered = complaints[complaints["car_id"].isin(tmp_join["car_id"].tolist())]

        #     audio_pz = MyDataset(id="audio", car_df=audio_filtered, schema=data_audio_cols)
        #     audio_pz = audio_pz.sem_filter("You are given an audio recording of car diagnostics. Return true if the recording captures an audio of a damaged car.", depends_on=["audio_path"])
        #     output_audio = audio_pz.run(copy.deepcopy(pz_config))
        #     audio_cost = output_audio.execution_stats.total_execution_cost
        #     output_audio = output_audio.to_df()
        #     output_audio["sick_audio"] = 1
            
        #     car_images_pz = MyDataset(id="car_images", car_df=car_images_filtered, schema=data_image_cols)
        #     car_images_pz = car_images_pz.sem_filter("You are given an image of a vehicle or its parts. Return true if car is damaged.", depends_on=["image_path"])
        #     output_images = car_images_pz.run(copy.deepcopy(pz_config))
        #     image_cost = output_images.execution_stats.total_execution_cost
        #     output_images = output_images.to_df()
        #     output_images["sick_image"] = 1

        #     complaints_pz = pz.MemoryDataset(id="complaints", vals=complaints_filtered)
        #     complaints_pz = complaints_pz.sem_filter("You are be given a textual complaint entailing that the car was on fire or burned. Complaint:", depends_on=["summary"])
        #     output_text = complaints_pz.run(copy.deepcopy(pz_config))
        #     text_cost = output_text.execution_stats.total_execution_cost
        #     output_text = output_text.to_df()
        #     output_text["sick_text"] = 1

        #     tmp_join.reset_index(inplace=True)

        #     res = tmp_join.join(output_audio.set_index('car_id'), on='car_id', how='outer', rsuffix="_au")
        #     res = res.join(output_images.set_index('car_id'), on='car_id', how='outer', rsuffix="_im")
        #     res = res.join(output_text.set_index('car_id'), on='car_id', how='outer', rsuffix="_txt")

        #     res["sick_text"] = np.where(res["summary_txt"].isna() & res["sick_text"].isna(), 0, res["sick_text"])
        #     res["sick_image"] = np.where(res["image_path_im"].isna() & res["sick_image"].isna(), 0, res["sick_image"])
        #     res["sick_audio"] = np.where(res["audio_path_au"].isna() & res["sick_audio"].isna(), 0, res["sick_audio"])

        #     # Find sick and healthy according to two different modalities
        #     res = res[((
        #     (res["sick_audio"] == 1) | (res["sick_text"] == 1) | (res["sick_image"] == 1)) & 
        #     ((res["sick_audio"] == 0) | (res["sick_text"] == 0) | (res["sick_image"] == 0)
        #     ))]

        #     cost = audio_cost + text_cost + image_cost

        #     return (res[["car_id", "year", "complaint_id", "image_id", "audio_id"]], cost)

    def _execute_q7(self) -> tuple[list[dict], dict]:
        """Q7: Find cars that are either dented (image), have worn out brakes (audio), or have electrical system problems (text), i.e., damaged at least according to a single modality."""
        # TODO: come back once carnot supports image and audio data
        return [], {"total_tokens": 0, "total_execution_cost": 0.0}

        # data_audio_cols = [
        #     {"name": "car_id", "type": int, "desc": "The integer id for the car"},
        #     {"name": "audio_path", "type": AudioFilepath, "desc": "The filepath containing audios."}
        # ]

        # data_image_cols = [
        #     {"name": "image_id", "type": int, "desc": "The integer id for the car image"},
        #     {"name": "car_id", "type": int, "desc": "The integer id for the car"},
        #     {"name": "image_path", "type": ImageFilepath, "desc": "The filepath containing the car image"},
        # ]


        # class MyDataset(pz.IterDataset):
        # def __init__(self, id: str, car_df: pd.DataFrame, schema: list):
        #     super().__init__(id=id, schema=schema)
        #     self.car_df = car_df

        # def __len__(self):
        #     return len(self.car_df)

        # def __getitem__(self, idx: int):
        #     # get row from dataframe
        #     return self.car_df.iloc[idx].to_dict()
        

        # def run(pz_config, data_dir: str, scale_factor: int = 157376):
        #     # Load data
        #     cars = pd.read_csv(os.path.join(data_dir, f"data/sf_{scale_factor}/car_data_{scale_factor}.csv")) 
        #     car_images = pd.read_csv(os.path.join(data_dir, f"data/sf_{scale_factor}/image_car_data_{scale_factor}.csv"))
        #     audio = pd.read_csv(os.path.join(data_dir, f"data/sf_{scale_factor}/audio_car_data_{scale_factor}.csv"))
        #     complaints = pd.read_csv(os.path.join(data_dir, f"data/sf_{scale_factor}/text_complaints_data_{scale_factor}.csv"))

        #     # Join for audio
        #     audio_join = cars.join(audio.set_index('car_id'), on='car_id', how='inner')[['car_id', 'year', 'audio_path']]
        #     audio_pz = MyDataset(id="audio", car_df=audio_join, schema=data_audio_cols)
        #     audio_pz = audio_pz.sem_filter("You are given an audio recording of car diagnostics. Return true if the car from the recording has worn out brakes.", depends_on=["audio_path"])
        #     output_audio = audio_pz.run(copy.deepcopy(pz_config))
        #     audio_cost = output_audio.execution_stats.total_execution_cost
        #     output_audio = output_audio.to_df()
            
        #     # Join for text
        #     text_join = cars.join(complaints.set_index('car_id'), on='car_id', how='inner')
        #     complaints_pz = pz.MemoryDataset(id="complaints", vals=text_join)
        #     complaints_pz = complaints_pz.sem_filter("In the complaint, the car has some problems with electrical system / connected to electrical system. Complaint:", depends_on=["summary"])
        #     output_text = complaints_pz.run(copy.deepcopy(pz_config))
        #     text_cost = output_text.execution_stats.total_execution_cost
        #     output_text = output_text.to_df()

        #     # Join for image
        #     image_join = cars.join(car_images.set_index('car_id'), on='car_id', how='inner')[['car_id', 'year', 'image_path', 'image_id']]
        #     car_images_pz = MyDataset(id="car_images", car_df=image_join, schema=data_image_cols)
        #     car_images_pz = car_images_pz.sem_filter("You are given an image of a vehicle or its parts. Return true if car is dented.", depends_on=["image_path"])
        #     output_images = car_images_pz.run(copy.deepcopy(pz_config))
        #     image_cost = output_images.execution_stats.total_execution_cost
        #     output_images = output_images.to_df()

        #     res = pd.concat([output_audio[['car_id']], output_text[['car_id']], output_images[['car_id']]], ignore_index=True).drop_duplicates()
        #     # Ensure column names are strings
        #     res.columns = [str(col) for col in res.columns]

        #     cost = audio_cost + text_cost + image_cost
        #     return (res, cost)

    def _execute_q8(self) -> tuple[list[dict], dict]:
        """Find a hundred cars with punctures and paint scratches on images."""
        # TODO: come back once carnot supports image data
        return [], {"total_tokens": 0, "total_execution_cost": 0.0}
    
        # # define the schema for the data you read from the file
        # car_image_data_cols = [
        #     {"name": "image_path", "type": ImageFilepath, "desc": "The filepath containing the car image"},
        #     {"name": "car_id", "type": int, "desc": "The integer id for the car"},
        #     {"name": "image_id", "type": int, "desc": "The integer id for the car image"},
        # ]


        # class MyDataset(pz.IterDataset):
        # def __init__(self, id: str, car_df: pd.DataFrame):
        #     super().__init__(id=id, schema=car_image_data_cols)
        #     self.car_df = car_df

        # def __len__(self):
        #     return len(self.car_df)

        # def __getitem__(self, idx: int):
        #     # get row from dataframe
        #     return self.car_df.iloc[idx].to_dict()
        

        # def run(pz_config, data_dir: str, scale_factor: int = 157376):
        #     # Load data
        #     car_images = pd.read_csv(os.path.join(data_dir, f"data/sf_{scale_factor}/image_car_data_{scale_factor}.csv")) 
        #     cars = pd.read_csv(os.path.join(data_dir, f"data/sf_{scale_factor}/car_data_{scale_factor}.csv")) 
            
        #     # Join before since PZ does not support joins
        #     tmp_join = cars.join(car_images.set_index('car_id'), on='car_id', how='inner')[['image_path', 'car_id', 'image_id']]

        #     tmp_join = MyDataset(id="my-car-data", car_df=tmp_join)

        #     # Filter image
        #     tmp_join = tmp_join.sem_filter('You are given an image of a vehicle or its parts. Return true if car has both, puncture and paint scratches.', depends_on=['image_path'])
        #     tmp_join = tmp_join.project(['car_id'])

        #     output = tmp_join.run(pz_config)
            
        #     # Handle limit in post-processing to avoid LimitScanOp hang
        #     result_df = output.to_df().head(100)
        #     result_df.columns = [str(col) for col in result_df.columns]
            
        #     # Return as wrapper to preserve execution stats
        #     class ResultWrapper:
        #         def __init__(self, df, exec_stats):
        #             self._df = df
        #             self.execution_stats = exec_stats
                
        #         def to_df(self):
        #             return self._df
            
        #     return ResultWrapper(result_df, output.execution_stats)

    def _execute_q9(self) -> tuple[list[dict], dict]:
        """Q9: Find cars that are torn according to images and have bad ignition according to audio."""
        # TODO: come back once carnot supports image and audio data
        return [], {"total_tokens": 0, "total_execution_cost": 0.0}

        # # define the schema for the data you read from the file
        # data_image_cols = [
        #     {"name": "image_id", "type": int, "desc": "The integer id for the car image"},
        #     {"name": "car_id", "type": int, "desc": "The integer id for the car"},
        #     {"name": "image_path", "type": ImageFilepath, "desc": "The filepath containing the car image"},
        # ]

        # data_audio_cols = [
        #     {"name": "car_id", "type": int, "desc": "The integer id for the car"},
        #     {"name": "audio_path", "type": AudioFilepath, "desc": "The filepath containing audios."},
        # ]


        # class MyDataset(pz.IterDataset):
        # def __init__(self, id: str, car_df: pd.DataFrame, schema: list):
        #     super().__init__(id=id, schema=schema)
        #     self.car_df = car_df

        # def __len__(self):
        #     return len(self.car_df)

        # def __getitem__(self, idx: int):
        #     # get row from dataframe
        #     return self.car_df.iloc[idx].to_dict()
        

        # def run(pz_config, data_dir: str, scale_factor: int = 157376):
        #     # Load data
        #     car_images = pd.read_csv(os.path.join(data_dir, f"data/sf_{scale_factor}/image_car_data_{scale_factor}.csv")) 
        #     cars = pd.read_csv(os.path.join(data_dir, f"data/sf_{scale_factor}/car_data_{scale_factor}.csv")) 
        #     audio = pd.read_csv(os.path.join(data_dir, f"data/sf_{scale_factor}/audio_car_data_{scale_factor}.csv")) 
            
        #     # Join cars with images
        #     image_join = cars.join(car_images.set_index('car_id'), on='car_id', how='inner')[['car_id', 'image_path', 'image_id']]
        #     image_pz = MyDataset(id="car_images", car_df=image_join, schema=data_image_cols)
        #     # Filter for torn cars
        #     image_pz = image_pz.sem_filter("You are given an image of a vehicle or its parts. Return true if car is torn according to the image.", depends_on=['image_path'])
        #     output_images = image_pz.run(copy.deepcopy(pz_config))
        #     image_cost = output_images.execution_stats.total_execution_cost
        #     output_images = output_images.to_df()
            
        #     # Join cars with audio
        #     audio_join = cars.join(audio.set_index('car_id'), on='car_id', how='inner')[['car_id', 'audio_path']]
        #     audio_pz = MyDataset(id="audio", car_df=audio_join, schema=data_audio_cols)
        #     # Filter for bad ignition
        #     audio_pz = audio_pz.sem_filter("You are given an audio recording of car diagnostics. Return true if the car from the recording has bad ignition.", depends_on=['audio_path'])
        #     output_audio = audio_pz.run(copy.deepcopy(pz_config))
        #     audio_cost = output_audio.execution_stats.total_execution_cost
        #     output_audio = output_audio.to_df()
            
        #     # Find cars that satisfy both conditions (torn AND bad ignition)
        #     torn_cars = set(output_images['car_id'].unique())
        #     bad_ignition_cars = set(output_audio['car_id'].unique())
        #     result_cars = torn_cars & bad_ignition_cars
            
        #     # Create result DataFrame
        #     result_df = pd.DataFrame({'car_id': list(result_cars)})
        #     result_df = result_df.drop_duplicates()
        #     # Ensure column names are strings
        #     result_df.columns = [str(col) for col in result_df.columns]
            
        #     # Return as tuple (DataFrame, cost) like Q7
        #     total_cost = image_cost + audio_cost
        #     return (result_df, total_cost)

    def _execute_q10(self) -> tuple[list[dict], dict]:
        """Q10: For all complaints, classify which car component is problematic according to the complaint."""
        # Load data
        complaints_df = self.load_data(f"text_complaints_data_{self.scale_factor}.csv")
        cars_df = self.load_data(f"car_data_{self.scale_factor}.csv")
        complaints_dataset = carnot.Dataset(
            name="Complaints Data",
            annotation="",
            items=complaints_df.to_dict(orient="records"),
        )
        cars_dataset = carnot.Dataset(
            name="Cars Data",
            annotation="",
            items=cars_df.to_dict(orient="records"),
        )
        execution = carnot.Execution(
            query="Classify which car component is problematic according to the complaint. The possible categories are: ELECTRICAL SYSTEM, POWER TRAIN, ENGINE, STEERING, SERVICE BRAKES, STRUCTURE, AIR BAGS, ENGINE AND ENGINE COOLING, VEHICLE SPEED CONTROL, VISIBILITY/WIPER, FUEL/PROPULSION SYSTEM, FORWARD COLLISION AVOIDANCE, EXTERIOR LIGHTING, SUSPENSION, FUEL SYSTEM, VISIBILITY, WHEELS, SEAT BELTS, BACK OVER PREVENTION, TIRES, SEATS, LATCHES/LOCKS/LINKAGES, LANE DEPARTURE, EQUIPMENT. For each complaint return the car_id and the problem_category.",
            datasets=[complaints_dataset, cars_dataset],
            llm_config=self.llm_config,
        )
        _, plan = execution.plan()
        execution._plan = plan
        output, _ = execution.run()

        return output, {"total_tokens": 0, "total_execution_cost": 0.0}
