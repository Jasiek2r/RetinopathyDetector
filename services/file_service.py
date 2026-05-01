
class FileService:
    def __init__(self, path):
        self.path = path

    def get_directory_path(self) -> str:
        return self.path


