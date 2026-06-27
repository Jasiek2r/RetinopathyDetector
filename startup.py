from core.retinopathy_application import RetinopathyApplication
from ml.concrete.model_provider import ModelProvider
from ml.concrete.retinopathy_model_loader import RetinopathyModelLoader
from ml.concrete.retinopathy_pipeline import RetinopathyPipeline
from services.classifier_service import ClassifierService
from services.file_service import FileService

from ml.concrete.retinopathy_ml_engine import RetinopathyMLEngine


class Startup:
    def __init__(self):
        self.__application__ = None

    def __build_application__(self) -> RetinopathyApplication:
        # configure constants here
        sample_limit = None  # use None for no limit
        path = "./augmented_resized_V2"

        # configure the dependencies used by application here
        provider = ModelProvider()
        engine = RetinopathyMLEngine(provider=provider)
        model_loader = RetinopathyModelLoader(engine=engine, provider=provider)
        pipeline = RetinopathyPipeline()
        classifier = ClassifierService(
            engine=engine,
            pipeline=pipeline
        )
        classifier.set_sample_limit(sample_limit)
        file_service = FileService(
            path=path
        )
        application = RetinopathyApplication(
            classifier_service=classifier,
            file_service=file_service,
            model_loader=model_loader
        )
        return application

    def run_application(self) -> None:
        self.__application__ = self.__build_application__()
        self.__application__.run()
