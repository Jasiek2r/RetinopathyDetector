from ml.abstractions.data_pipeline import DataPipeline
from ml.abstractions.ml_engine import MLEngine
from utility.decorated_print import print_decorated
from utility.formatted_date import get_formatted_date


class ClassifierService:
    def __init__(self, engine: MLEngine, pipeline: DataPipeline):
        self.sample_limit = None
        self.__engine__ = engine
        self.__pipeline__ = pipeline

    def set_sample_limit(self, sample_limit):
        self.sample_limit = sample_limit

    def train(self, dir_path: str):

        train_dataset, val_dataset, test_dataset = self.__pipeline__.run(
            dir_path=dir_path,
            max_images=self.sample_limit
        )

        print("Images have been loaded successfully")
        print_decorated("RETINOPATHY DIAGNOSTIC APPLICATION")

        formatted_date = get_formatted_date()
        print(f"Training started at {formatted_date}")

        # --- TRENING ---
        self.__engine__.train(train_dataset, val_dataset)

        formatted_date = get_formatted_date()
        print(f"Training finished at {formatted_date}")

        # --- TEST ---
        print("Evaluating on test split...")
        self.__engine__.test(test_dataset)