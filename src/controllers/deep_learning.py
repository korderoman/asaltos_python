import os


class DeepLearningController:
    def __init__(self):
        self.input_video_path = "../../../backend/data"
        self.output_video_local_path = "../data"

    def check_if_video_exists(self, id_video):
        full_path = f"{self.input_video_path}/{id_video}.mp4"
        if os.path.exists(full_path):
            return True
        return False

    def pipeline(self, id_video):
        video_exist = self.check_if_video_exists(id_video)
        if video_exist:
            # el pipeline debe de guardar el video dentro de data y agregar un sufijo
            # al nombre del video que se llame learned
            local_path = f"{self.output_video_local_path}/{id_video}_learned.mp4"
            return {"success": True, "message": "El vídeo ha sido procesado con éxito",
                    "extras": {"local_path": local_path}}
        return {"success": False, "message": "El video no existe o no ha sido posible procesarlo"}
