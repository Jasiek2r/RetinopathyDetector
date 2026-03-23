from ml.concrete.research_project.research_project_dataset import ResearchProjectDataset
from services.abstractions.loader_service import LoaderService


class ResearchProjectLoaderService(LoaderService):

    def load_data(self, dataset_dir: str, max_images=None):
        return ResearchProjectDataset(dataset_dir, max_images)
