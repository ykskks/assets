import io
import os

from google.colab import auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

auth.authenticate_user()

drive_service = build("drive", "v3")
results = (
    drive_service.files().list(q="name = 'kaggle.json'", fields="files(id)").execute()
)
kaggle_api_key = results.get("files", [])

filename = "/content/.kaggle/kaggle.json"
os.makedirs(os.path.dirname(filename), exist_ok=True)

request = drive_service.files().get_media(fileId=kaggle_api_key[0]["id"])
fh = io.FileIO(filename, "wb")
downloader = MediaIoBaseDownload(fh, request)
done = False
while done is False:
    status, done = downloader.next_chunk()
    print("Download %d%%." % int(status.progress() * 100))
os.chmod(filename, 600)

# !mkdir /content/drive/MyDrive/kaggle/commonlit/upload/

# !mv /content/drive/MyDrive/kaggle/commonlit/output/model0.bin /content/drive/MyDrive/kaggle/commonlit/upload/
# !mv /content/drive/MyDrive/kaggle/commonlit/output/model1.bin /content/drive/MyDrive/kaggle/commonlit/upload/
# !mv /content/drive/MyDrive/kaggle/commonlit/output/model2.bin /content/drive/MyDrive/kaggle/commonlit/upload/
# !mv /content/drive/MyDrive/kaggle/commonlit/output/model3.bin /content/drive/MyDrive/kaggle/commonlit/upload/
# !mv /content/drive/MyDrive/kaggle/commonlit/output/model4.bin /content/drive/MyDrive/kaggle/commonlit/upload/

# !cp -r /content/drive/MyDrive/kaggle/.kaggle /root/

# !kaggle datasets init -p /content/drive/MyDrive/kaggle/commonlit/upload


dataset_metadata = {}
dataset_metadata["id"] = "ykskks/roberta-base-v18"
dataset_metadata["licenses"] = [{"name": "CC0-1.0"}]
dataset_metadata["title"] = "roberta-base-v18"

with open(
    "/content/drive/MyDrive/kaggle/commonlit/upload/dataset-metadata.json", "w"
) as f:
    json.dump(dataset_metadata, f, indent=4)

# !kaggle datasets create -p /content/drive/MyDrive/kaggle/commonlit/upload
# !rm -rf /content/drive/MyDrive/kaggle/commonlit/upload
