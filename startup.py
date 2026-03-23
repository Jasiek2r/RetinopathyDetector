from core.retinopathy_application import RetinopathyApplication
from services.classifier_service import ClassifierService
from services.concrete.research_project.research_project_loader_service import ResearchProjectLoaderService
from services.file_service import FileService

from ml.concrete.research_project.research_project_ml_engine import ResearchProjectMLEngine

class Startup:
    def __init__(self):
        self.__application__ = None

    def __build_application__(self) -> RetinopathyApplication:
        # configure constants here
        sample_limit = 1000  # use None for no limit
        path = "C:\\Users\\janek\\Desktop\\STUDIA\\magisterskie\\semestr 1\\szum\\projekt\\aptos2019-blindness-detection"
        dev = True  # uses developer version for testing

        # configure the dependencies used by application here
        engine = ResearchProjectMLEngine()
        loader_service = ResearchProjectLoaderService()
        classifier = ClassifierService(
            engine=engine,
            loader_service=loader_service
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
