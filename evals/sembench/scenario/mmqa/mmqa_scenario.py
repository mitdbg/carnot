import glob
import os

MMQA_FILES_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "files", "mmqa")
)


class MMQAScenario:
    """
    MMQA scenario handler.

    This class:
     * downloads and prepares data
     * retrieves queries
    """

    def __init__(self, scale_factor: int = 25):
        self.data_dir = None
        self.scale_factor = scale_factor

    def setup_scenario(self, systems: list[str]) -> None:
        # Download and prepare data if not already done
        from pathlib import Path

        from scenario.mmqa.preparation.generate_data import MMQADataGenerator

        # Get the working directory (base directory of the project)
        working_dir = Path(__file__).resolve().parents[2]

        # Check if data already exists
        data_folder = Path(MMQA_FILES_DIR) / "data" / f"sf_{self.scale_factor}"

        if data_folder.exists() and len(list(data_folder.glob("*.csv"))) > 0:
            print(f"Data already exists at {data_folder}, skipping generation.")
            self.data_dir = str(data_folder)
        else:
            # Generate data
            data_generator = MMQADataGenerator(
                working_dir=str(working_dir),
                scale_factor=self.scale_factor,
                skip_download=False,
            )
            data_generator.generate_data()
            self.data_dir = data_generator.output_data_dir

        # Load data into the specified systems
        for system in systems:
            if system == "bigquery" or system == "flockmtl":
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

    def get_query_text(self, query_id, system_name: str) -> str:
        """
        Get the SQL query text for a given query ID and system name.

        Args:
            query_id: ID of the query
            system_name: Name of the system
        Returns:
            SQL query text as a string
        """
        system_query_dir = os.path.abspath(
            os.path.join(MMQA_FILES_DIR, "query", system_name)
        )
        matching_files = glob.glob(os.path.join(system_query_dir, f"q{query_id}.*"))

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
                os.path.join(MMQA_FILES_DIR, "data", f"sf_{self.scale_factor}")
            )

    def discover_available_queries(self, system_name: str = None) -> list[int]:
        """
        Discover available queries for the MMQA scenario.

        Returns:
            List of query IDs.
        """
        if system_name is None:
            # Return all possible queries from lotus directory
            system_query_dir = os.path.abspath(
                os.path.join(MMQA_FILES_DIR, "query", "lotus")
            )
        else:
            system_query_dir = os.path.abspath(
                os.path.join(MMQA_FILES_DIR, "query", system_name)
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
