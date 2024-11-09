import os
import  platform
from dotenv import load_dotenv
from src.pipeline.pipeline import PipelineController
from werkzeug.utils import secure_filename
import subprocess


load_dotenv()


class DeepLearningController:
    def __init__(self):
        self.input_video_path = "../../../backend/data"
        self.output_video_local_path = "../data"
        self.pipelineController = PipelineController()
    def get_input_local_path_mp4(self, id_video):
        directory = os.getenv("DIRECTORY_JS_APP")
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..", directory, "data", id_video + ".mp4"))

    def check_if_video_exists(self, id_video):
        full_path = self.get_input_local_path_mp4(id_video)
        if os.path.exists(full_path):
            return True
        return False

    def pipeline(self, id_video):
        video_exist = self.check_if_video_exists(id_video=id_video)
        if video_exist:
            # el pipeline debe de guardar el video dentro de data y agregar un sufijo
            # al nombre del video que se llame learned
            input_path_mp4 = self.get_input_local_path_mp4(id_video=id_video)
            output_path_mp4 = self.get_output_local_path_mp4(id_video=id_video)
            input_path_avi=self.pass_from_mp4_to_avi(input_path_mp4=input_path_mp4, id_video=id_video)
            output_path_avi=self.pass_to_output_path_avi_processed(output_path=input_path_avi)
            print("data",output_path_avi, input_path_avi)
            prediction=self.pipelineController.predecir_video(input_path_mp4, output_path_mp4)
            return {"success": True, "message": "El vídeo ha sido procesado con éxito",
                    "extras": {"local_path": output_path_avi, "name_file": f"{id_video}_learned.avi","has_violence":self.has_violence(prediction)}}
        return {"success": False, "message": "El video no existe o no ha sido posible procesarlo"}

    def pass_to_output_path_avi_processed(self, output_path):
        return output_path.rsplit(".", 1)[0] + "_learned.avi"
    def has_violence(self, prediction):
        if prediction==1:
            return True
        return False
    def get_output_local_path_mp4(self, id_video):
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", id_video + "_learned.mp4"))
    def get_output_local_path_avi(self, id_video):
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", id_video + ".avi"))
    def pass_from_mp4_to_avi(self, input_path_mp4, id_video):
        print("buscando el archivo para el pase de mp4 a avi: ",input_path_mp4 )
        output_filename_avi=f"{id_video}_learned.avi"
        output_filepath_avi = self.get_output_local_path_avi(id_video=id_video)
        ffmpeg_path="ffmpeg"
        if platform.system() == "Windows":
            ffmpeg_path = r"C:\ffmpeg\bin\ffmpeg.exe"
        ffmpeg_command = [
           ffmpeg_path, '-i', input_path_mp4, '-c:v', 'libxvid', '-qscale:v', '5','-c:a', 'libmp3lame', '-qscale:a', '4','-y',
          output_filepath_avi
        ]
        #ffmpeg_command = [
         #   ffmpeg_path, "-y", "-i", input_path_mp4, "-c:v", "libxvid", "-b:v", "500k",
         #   "-c:a", "libmp3lame", "-b:a", "128k", output_filepath_avi
        #]
        # Ruta completa al ejecutable de FFmpeg
        try:
            # Ejecutar el comando de FFmpeg
            subprocess.run(ffmpeg_command, check=True)
            print("Conversión completada con éxito.")
            return output_filepath_avi
        except subprocess.CalledProcessError as e:
            print(f"Error durante la conversión: {e}")
        except FileNotFoundError:
            print("FFmpeg no está instalado o no se encuentra en el PATH.")
        except Exception as e:
            print(f"Ha ocurrido un error inesperado: {e}")


