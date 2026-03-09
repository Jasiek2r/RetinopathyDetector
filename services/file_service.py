import tkinter as tk
from tkinter import filedialog


class FileService:
    def __init__(self, is_dev, path):
        self.is_dev = is_dev
        self.path = path

    def get_directory_path(self) -> str:
        if self.is_dev:
            return self.path
        root = tk.Tk()
        root.withdraw()
        return filedialog.askdirectory()
