from google.oauth2 import  service_account
from google.cloud import storage
import os
from dotenv import load_dotenv
load_dotenv()
class CloudStorage:
    def __init__(self):
        self.credentials = service_account.Credentials.from_service_account_file(os.getenv("SERVICE_ACCOUNT"))
        self.client=storage.client.Client(project=os.getenv("PROJECT_ID"), credentials=self.credentials)
        self.bucket = self.client.get_bucket(os.getenv("BUCKET_NAME"))

    def upload_to_cloud_storage(self, data):
        try:
            filename = data["extras"]["name_file"]
            local_path=data["extras"]["local_path"]
            blob=self.bucket.blob(filename)
            blob.upload_from_filename(local_path)
            url = blob.public_url
            return {"status":"success", "message":"Se subió con éxito el archivo de aprendizaje", "extras" :{"url":url}}
        except Exception as e:
            print(e)
            return {"status":"error", "message":"el archivo tuvo un error en el proceso de subida"}

    def check_if_file_exists(self, filename):
        full_path=os.path.join(os.path.dirname(__file__), filename)
        print(full_path)