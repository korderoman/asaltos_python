from flask import Blueprint, request
from src.controllers import DeepLearningController, CloudStorage

router = Blueprint('router', __name__)
deep_controller = DeepLearningController()
cloud_storage_controller = CloudStorage()


@router.route('/', methods=['GET'])
def index():
    return "<p>Flask running</p>"


@router.route('/process', methods=['POST'])
def process():
    if request.method == 'POST':
        data = request.get_json()
        response1 = deep_controller.pipeline(id_video=data['video_id'])
        print(response1)
        response2=cloud_storage_controller.upload_to_cloud_storage(data=response1)
        return response2
