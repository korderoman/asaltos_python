from src.pipeline.extractor import Extractor
from src.pipeline.transformer_model import TransformerModel
import torch
import cv2
import torchvision.transforms as transforms
import subprocess
import os


class PipelineController:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = Extractor(self.device)
        self.extractor.to(self.device)
        self.transform = self.get_transform()
        self.dropout = 0.5
        self.num_features = 512
        self.num_classes = 2
        self.model_path=self.get_model_path()
        self.model = self.load_model()

    def get_transform(self):
        return transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def get_model_path(self):
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "model", "modelo_transformer_violence_detection.pth"))
    def load_model(self):
        model=TransformerModel(num_features=self.num_features, num_classes=self.num_classes,
                                      dropout=self.dropout)
        model.load_state_dict(torch.load(self.model_path, map_location=self.device))
        model.to(self.device)
        return model

    def get_output_pre_processed_path(self):
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "data","pre_procesado.avi"))

    def predecir_video(self, input_path, output_path):
        print("Iniciando la predicción")
        # Pre_procesar el video antes de extraer características
        pre_procesado_path = self.get_output_pre_processed_path()
        self.pre_procesar_y_guardar_video(input_path, output_path)
        # Extraer las características del video pre_procesado
        caracteristicas = self.extraer_caracteristicas(output_path, self.extractor)

        # Asegurarse de que las características estén en el dispositivo correcto
        caracteristicas = caracteristicas.unsqueeze(0).to(self.device)  # Añadir dimensión de batch

        # Poner el modelo en modo de evaluación
        self.model.eval()
        # Realizar la predicción
        with torch.no_grad():
            salida = self.model(caracteristicas)
            _, prediccion = torch.max(salida, 1)

        # Interpretar la predicción
        if prediccion.item() == 1:
            print(f"El video {os.path.basename(input_path)} contiene violencia.")
        else:
            print(f"El video {os.path.basename(input_path)} no contiene violencia.")

        return prediccion.item()

    def pre_procesar_y_guardar_video(self, video_path, output_path, target_width=224, target_height=224, target_fps=15,
                                     to_gray=True):
        print("Iniciando el pre procesamiento")
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error al abrir el video: {video_path}")
                return

            original_fps = cap.get(cv2.CAP_PROP_FPS)
            skip_rate = int(original_fps / target_fps) if original_fps > target_fps else 1

            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            #fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            is_color = not to_gray
            out = cv2.VideoWriter(output_path, fourcc, target_fps, (target_width, target_height), isColor=is_color)

            frame_count = 0
            processed_frames = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % skip_rate == 0:
                    frame = cv2.resize(frame, (target_width, target_height))
                    if to_gray:
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    out.write(frame if to_gray else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
                    processed_frames += 1

                frame_count += 1

            cap.release()
            out.release()
        except Exception as e:
            print(f"Error durante el preprocesamiento {e}")

    def verificar_video_ffmpeg(self, video_path):
        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                 "stream=width,height,r_frame_rate,duration,nb_frames", "-of", "default=noprint_wrappers=1",
                 video_path],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            #

            output = result.stdout.decode()
            props = {}
            for line in output.splitlines():
                if "=" in line:
                    key, value = line.split("=")
                    if value != 'N/A':
                        if key in ["width", "height", "nb_frames"]:
                            props[key] = int(value)
                        elif key in ["r_frame_rate"]:
                            num, denom = map(int, value.split("/"))
                            props[key] = num / denom if denom != 0 else 0
                        elif key in ["duration"]:
                            props[key] = float(value)

            if "nb_frames" not in props and "duration" in props and "r_frame_rate" in props:
                props["nb_frames"] = int(props["duration"] * props["r_frame_rate"])

            print("Propiedades de verificación con FFmpeg:")
            print(f"Resolución: {props.get('width')} x {props.get('height')}")
            print(f"FPS: {props.get('r_frame_rate')}")
            print(f"Duración: {props.get('duration')} segundos")
            print(f"Número total de cuadros: {props.get('nb_frames')}")

        except Exception as e:
            print("Error al verificar las propiedades del video con FFmpeg:", e)

    def extraer_caracteristicas(self, video_path, extractor, batch_size=32):
        print("Iniciando la extracción de características")
        try:

            cap = cv2.VideoCapture(video_path)
            frames_features = []
            batch_frames = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    if len(batch_frames) > 0:  # Asegúrate de procesar el último lote
                        batch_tensor = torch.stack(batch_frames).to(self.device)
                        features = extractor(batch_tensor)
                        frames_features.extend(features.cpu())  # Guardar características a lista
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = self.transform(frame_rgb)
                batch_frames.append(frame_tensor)

                if len(batch_frames) == batch_size:
                    batch_tensor = torch.stack(batch_frames).to(self.device)
                    features = extractor(batch_tensor)
                    frames_features.extend(features.cpu())  # Guardar características a lista
                    batch_frames = []  # Reiniciar el lote

            cap.release()
            return torch.stack(frames_features)
        except Exception as e:
            print(f"Ha ocurrido un error al extraer las características {e}")
