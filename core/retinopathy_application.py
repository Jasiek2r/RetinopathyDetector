from ml.abstractions.model_loader import ModelLoader
from services.classifier_service import ClassifierService
from services.file_service import FileService
from utility.decorated_print import print_decorated
from utility.user_query import UserQuerer


class RetinopathyApplication:

    def __init__(self,
                 classifier_service: ClassifierService,
                 file_service: FileService,
                 model_loader: ModelLoader):
        self.__classifier_service__ = classifier_service
        self.__file_service__ = file_service
        self.__model_loader__ = model_loader
        self.__app_ready__ = False

    def run(self) -> None:
        querer = UserQuerer()
        while not self.__app_ready__:
            print_decorated("RETINOPATHY DIAGNOSTIC APPLICATION")
            query_result = querer.retrieve_input(headers=[
                "Would you like to start fresh or recover model from backup?",
                "- Enter F to start fresh",
                "- Enter M to load model from backup"
            ], permitted_values=[
                "F", "M"
            ])
            if query_result == "F":
                acceptance_result = querer.retrieve_acceptance(headers=[
                    "You are going to train the model shortly",
                    "After confirming you are going to be asked to select the folder with training data",
                    "Bear in mind that an application should NOT be turned off during training process"
                ])
                if acceptance_result == "Y":
                    print("Getting directory...")
                    training_data_directory = self.__file_service__.get_directory_path()
                    print("Loading images...")
                    self.__classifier_service__.train(training_data_directory)
                    self.__app_ready__ = True
            elif query_result == "M":
                self.__model_loader__.load(path=input("Please provide the path: "))
                test_data_directory = self.__file_service__.get_directory_path()
                self.__classifier_service__.full_evaluation(test_data_directory)


        input("Type anything to quit ")
