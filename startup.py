from core.retinopathy_application import RetinopathyApplication
from enums.project_type import ProjectType
from ml.concrete.SzUM.szum_project_ml_engine import SzumProjectMLEngine
from services.classifier_service import ClassifierService
from services.concrete.SzUM.szum_loader_service import SzumProjectLoaderService
from services.concrete.research_project.research_project_loader_service import ResearchProjectLoaderService
from services.file_service import FileService

from ml.concrete.research_project.research_project_ml_engine import ResearchProjectMLEngine

class Startup:
    def __init__(self, project_type: ProjectType):
        self.__application__ = None
        self.project_type = project_type

    def __build_research_project_application(self) -> RetinopathyApplication:
        # configure constants here
        sample_limit = 1000  # use None for no limit
        path = "C:\\Users\\janek\\Downloads\\aptos2019-blindness-detection"
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

    def __build_systems_with_machine_learning_application(self) -> RetinopathyApplication:
        # configure constants here
        sample_limit = 1000  # use None for no limit
        path = "C:\\Users\\janek\\Downloads\\aptos2019-filtered"
        dev = True  # uses developer version for testing

        # configure the dependencies used by application here
        engine = SzumProjectMLEngine()
        loader_service = SzumProjectLoaderService()
        classifier = ClassifierService(
            engine=engine,
            loader_service=loader_service
        )
        classifier.set_sample_limit(sample_limit)
        file_service = FileService(
            is_dev=dev,
            path=path
        )
        application = RetinopathyApplication(
            classifier_service=classifier,
            file_service=file_service
        )
        return application

    def __build_application__(self) -> RetinopathyApplication:
        if self.project_type == ProjectType.GROUP_RESEARCH_PROJECT:
            return self.__build_research_project_application()
        elif self.project_type == ProjectType.SYSTEMS_WITH_ML_PROJECT:
            return self.__build_systems_with_machine_learning_application()
        raise RuntimeError("Cannot start an app of specified project type")

    def run_application(self) -> None:
        self.__application__ = self.__build_application__()
        self.__application__.run()
