import glob
import os

from scenario.animals.preparation.generate_data import download_from_google_drive

ANIMALS_FILES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "files", "animals")
)


class AnimalsScenario:
    """
    Animals scenario handler.

    This class:
     * downloads and prepares data
     * retrieves queries
    """

    def __init__(self, scale_factor: int = 500):
        self.data_dir = None
        self.scale_factor = scale_factor

    def setup_scenario(self, systems: list[str]) -> None:
        # Download and prepare data if not already done
        import random
        from pathlib import Path

        from scenario.animals.preparation.generate_data import (
            _ensure_cooccurrence_patterns,
            _ensure_q6_pattern,
            _ensure_q9_pattern,
            _generate_audio_table,
            _generate_image_table,
        )

        # Set random seed for reproducibility
        random.seed(42)

        # Check if data already exists
        data_folder = Path(ANIMALS_FILES_DIR) / "data" / f"sf_{self.scale_factor}"
        audio_file = data_folder / "audio_data.csv"
        image_file = data_folder / "image_data.csv"

        if audio_file.exists() and image_file.exists():
            print(f"Data already exists at {data_folder}, skipping generation.")
            self.data_dir = str(data_folder)
        else:
            # Download source data
            audio_path, image_path = download_from_google_drive()

            # Calculate table sizes
            max_audio_files = 650
            max_image_files = 8718
            audio_size = min(self.scale_factor // 3, max_audio_files)
            image_size = min(self.scale_factor, max_image_files)

            print(f"Generating tables: Audio={audio_size}, Image={image_size}")

            # Generate tables
            audio_table = _generate_audio_table(audio_path, audio_size)
            image_table = _generate_image_table(image_path, image_size)

            # Ensure co-occurrence patterns
            audio_table, image_table = _ensure_cooccurrence_patterns(
                audio_table, image_table
            )
            audio_table, image_table = _ensure_q9_pattern(audio_table, image_table)
            audio_table, image_table = _ensure_q6_pattern(audio_table, image_table)

            # Save to data directory
            os.makedirs(data_folder, exist_ok=True)
            audio_table.to_csv(audio_file, index=False)
            image_table.to_csv(image_file, index=False)

            print(f"Data saved to {data_folder}")
            self.data_dir = str(data_folder)

        # Load data into the specified systems
        for system in systems:
            if system == "bigquery":
                pass
            elif system == "lotus":
                pass  # Nothing to do. LOTUS works on raw files.
            elif system == "palimpzest":
                pass  # Nothing to do. Palimpzest works on raw files.
            elif system == "carnot":
                pass  # Nothing to do. Carnot works on raw files.
            elif system == "thalamusdb":
                pass  # Nothing to do. ThalamusDB works on raw files.
            else:
                raise ValueError(f"Unsupported system: {system}")

    def get_query_text(self, query_id: int, system_name: str) -> str:
        """
        Get the SQL query text for a given query ID and system name.

        Args:
            query_id: ID of the query
            system_name: Name of the system
        Returns:
            SQL query text as a string
        """
        system_query_dir = os.path.abspath(
            os.path.join(ANIMALS_FILES_DIR, "query", system_name)
        )
        matching_files = glob.glob(os.path.join(system_query_dir, f"Q{query_id}.*"))

        if not matching_files:
            raise FileNotFoundError(
                f"No query implementation found for query {query_id} and system '{system_name}'"
            )
        if len(matching_files) > 1:
            raise ValueError(
                f"Multiple query files found for query {query_id} and system '{system_name}': {matching_files}"
            )

        with open(matching_files[0]) as f:
            return f.read()

    def get_data_dir(self) -> str:
        if self.data_dir:
            return self.data_dir
        else:
            return os.path.abspath(
                os.path.join(ANIMALS_FILES_DIR, "data", f"sf_{self.scale_factor}")
            )

    def discover_available_queries(self, system_name: str = None) -> list[int]:
        """
        Discover available queries for the animals scenario.

        Returns:
            List of query IDs.
        """
        if system_name is None:
            # Return all possible queries
            system_query_dir = os.path.abspath(
                os.path.join(ANIMALS_FILES_DIR, "query", "lotus")
            )
        else:
            system_query_dir = os.path.abspath(
                os.path.join(ANIMALS_FILES_DIR, "query", system_name)
            )

        if not os.path.exists(system_query_dir):
            return []

        query_files = glob.glob(os.path.join(system_query_dir, "Q*.*"))
        query_ids = []
        for f in query_files:
            try:
                # Extract number from filename like Q1.py or Q1.sql
                filename = os.path.basename(f)
                query_id = int(filename[1:].split(".")[0])
                query_ids.append(query_id)
            except ValueError:
                continue
        return sorted(query_ids)
