from ml.abstractions.ml_engine import MLEngine
from services.abstractions.loader_service import LoaderService
from services.concrete.research_project.research_project_loader_service import ResearchProjectLoaderService
from utility.decorated_print import print_decorated
from utility.formatted_date import get_formatted_date


class ClassifierService:
    def __init__(self, engine: MLEngine, loader_service: LoaderService):
        self.sample_limit = None
        self.__engine__ = engine
        self.__loader_service__ = loader_service

    def set_sample_limit(self, sample_limit):
        self.sample_limit = sample_limit

    def train(self, dir_path: str):
        x, y = self.__loader_service__.load_data(dir_path, self.sample_limit)

        if self.sample_limit is not None:
            x = x[:self.sample_limit]
            y = y[:self.sample_limit]

        # --- PODZIAŁ DANYCH ---
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.2, random_state=42
        )

        print("Images have been loaded successfully")
        print_decorated("RETINOPATHY DIAGNOSTIC APPLICATION")

        formatted_date = get_formatted_date()
        print(f"Training started at {formatted_date}")
        self.__engine__.train(x_train, y_train)
        formatted_date = get_formatted_date()
        print(f"Training finished at {formatted_date}")

        # --- AUTOMATYCZNA EWALUACJA PO TRENINGU ---
        print("Evaluating on test split...")
        self.__engine__.test(x_test, y_test)
