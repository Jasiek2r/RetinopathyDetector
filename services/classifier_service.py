from ml.abstractions.data_pipeline import DataPipeline
from ml.abstractions.ml_engine import MLEngine
from ml.abstractions.zero_shot_engine import ZeroShotEngine
from utility.decorated_print import print_decorated
from utility.formatted_date import get_formatted_date


class ClassifierService:
    def __init__(self, engine: MLEngine, pipeline: DataPipeline, zero_shot_engine: ZeroShotEngine):
        self.sample_limit = None
        self.__engine__ = engine
        self.__pipeline__ = pipeline
        self.__zero_shot_engine__ = zero_shot_engine

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

    def full_evaluation(self, dir_path: str):
        print("Full debug evaluation started...")
        train_dataset, val_dataset, test_dataset = self.__pipeline__.run(
            dir_path=dir_path,
            max_images=self.sample_limit
        )
        self.__engine__.full_evaluation(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset
        )

    def zero_shot(self, dir_path: str):
        train_dataset, _, test_dataset = self.__pipeline__.run(
            dir_path=dir_path,
            max_images=self.sample_limit
        )
        self.__zero_shot_engine__.build_prototypes(train_dataset)
        self.__zero_shot_engine__.evaluate(test_dataset)
