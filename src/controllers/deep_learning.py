import os
from dotenv import load_dotenv
from src.pipeline.pipeline import PipelineController

load_dotenv()


class DeepLearningController:
    def __init__(self):
        self.input_video_path = "../../../backend/data"
        self.output_video_local_path = "../data"
        self.pipelineController = PipelineController()
    def get_input_local_path(self, id_video):
        directory = os.getenv("DIRECTORY_JS_APP")
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", directory, "data", id_video + ".mp4"))

    def check_if_video_exists(self, id_video):
        full_path = self.get_input_local_path(id_video)
        if os.path.exists(full_path):
            return True
        return False

    def pipeline(self, id_video):
        video_exist = self.check_if_video_exists(id_video=id_video)
        if video_exist:
            # el pipeline debe de guardar el video dentro de data y agregar un sufijo
            # al nombre del video que se llame learned
            input_path = self.get_input_local_path(id_video=id_video)
            output_path = self.get_output_local_path(id_video=id_video)
            prediction=self.pipelineController.predecir_video(input_path, output_path)
            return {"success": True, "message": "El vídeo ha sido procesado con éxito",
                    "extras": {"local_path": output_path, "name_file": f"{id_video}_learned.mp4","has_violence":self.has_violence(prediction)}}
        return {"success": False, "message": "El video no existe o no ha sido posible procesarlo"}
    def has_violence(self, prediction):
        if prediction==1:
            return True
        return False
    def get_output_local_path(self, id_video):
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", id_video + "_learned.mp4"))