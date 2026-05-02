from core.retinopathy_application import RetinopathyApplication
from services.classifier_service import ClassifierService
from services.concrete.research_project.retinopathy_loader_service import ResearchProjectLoaderService
from services.file_service import FileService

from ml.concrete.retinopathy_ml_engine import RetinopathyMLEngine

class Startup:
    def __init__(self):
        self.__application__ = None

    def __build_application__(self) -> RetinopathyApplication:
        # configure constants here
        sample_limit = 50  # use None for no limit
        path = "./split-1"
        dev = True  # uses developer version for testing

        # configure the dependencies used by application here
        engine = RetinopathyMLEngine()
        loader_service = ResearchProjectLoaderService()
        classifier = ClassifierService(
            engine=engine,
            loader_service=loader_service,
        )
        classifier.set_sample_limit(sample_limit)
        file_service = FileService(
            is_dev=dev,
            path=path
        )
        application = RetinopathyApplication(classifier, file_service)
        return application


    def run_application(self) -> None:
        self.__application__ = self.__build_application__()
        self.__application__.run()
