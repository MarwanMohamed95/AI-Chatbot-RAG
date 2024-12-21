import os

class BaseController:
    
    def __init__(self):
        
        self.base_dir = os.path.dirname( os.path.dirname(__file__) )
        self.files_dir = os.path.join(
            self.base_dir,
            "assets/files"
        )

    def get_file_extension(self, file_path: str):
        return os.path.splitext(file_path)[-1]
