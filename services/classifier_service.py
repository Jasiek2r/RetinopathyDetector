from ml.abstractions.data_pipeline import DataPipeline
from ml.abstractions.ml_engine import MLEngine
from ml.concrete.retinopathy_dataset import RetinopathyDataset
from services.abstractions.loader_service import LoaderService
from utility.decorated_print import print_decorated
from utility.formatted_date import get_formatted_date


class ClassifierService:
    def __init__(self, engine: MLEngine, loader_service: LoaderService, pipeline: DataPipeline):
        self.sample_limit = None
        self.__engine__ = engine
        self.__loader_service__ = loader_service
        self.__pipeline__ = pipeline

    def set_sample_limit(self, sample_limit):
        self.sample_limit = sample_limit

    def train(self, dir_path: str):
        dataset = self.__loader_service__.load_data(dir_path, self.sample_limit)

        print("Dataset size:", len(dataset))

        train_dataset, val_dataset, test_dataset = self.__pipeline__.run(
            data=dataset,
            path=dir_path
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
