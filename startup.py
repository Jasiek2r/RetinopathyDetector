from core.retinopathy_application import RetinopathyApplication
from ml.concrete.retinopathy_model_loader import RetinopathyModelLoader
from services.classifier_service import ClassifierService
from services.concrete.research_project.retinopathy_loader_service import ResearchProjectLoaderService
from services.file_service import FileService

from ml.concrete.retinopathy_ml_engine import RetinopathyMLEngine

class Startup:
    def __init__(self):
        self.__application__ = None

    def __build_application__(self) -> RetinopathyApplication:
        # configure constants here
        sample_limit = None  # use None for no limit
        path = "./split-2"
        dev = True  # uses developer version for testing

        # configure the dependencies used by application here
        engine = RetinopathyMLEngine()
        loader_service = ResearchProjectLoaderService()
        model_loader = RetinopathyModelLoader(engine=engine)
        classifier = ClassifierService(
            engine=engine,
            loader_service=loader_service,
        )
        classifier.set_sample_limit(sample_limit)
        file_service = FileService(
            is_dev=dev,
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
