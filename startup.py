from core.retinopathy_application import RetinopathyApplication
from ml.concrete.retinopathy_pipeline import RetinopathyPipeline
from services.classifier_service import ClassifierService
from services.concrete.research_project.retinopathy_loader_service import ResearchProjectLoaderService
from services.file_service import FileService

from ml.concrete.retinopathy_ml_engine import RetinopathyMLEngine

class Startup:
    def __init__(self):
        self.__application__ = None

    def __build_application__(self) -> RetinopathyApplication:
        # configure constants here
        sample_limit = 2000  # use None for no limit
        path = "./research-project-dataset"

        # configure the dependencies used by application here
        engine = RetinopathyMLEngine()
        loader_service = ResearchProjectLoaderService()
        pipeline = RetinopathyPipeline()
        classifier = ClassifierService(
            engine=engine,
            loader_service=loader_service,
            pipeline=pipeline
        )
        classifier.set_sample_limit(sample_limit)
        file_service = FileService(
            path=path
        )
        application = RetinopathyApplication(classifier, file_service)
        return application


    def run_application(self) -> None:
        self.__application__ = self.__build_application__()
        self.__application__.run()
