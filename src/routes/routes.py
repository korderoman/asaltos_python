from flask import Blueprint, request

router = Blueprint('router', __name__)


@router.route('/', methods=['GET'])
def index():
    return "<p>Flask running</p>"

@router.route('/process', methods=['POST'])
def process():
    if request.method == 'POST':
        data = request.get_json()
        print(data)
