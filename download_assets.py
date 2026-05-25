import os
import gdown
import zipfile

DATASET_PATH = "assets/text_to_sign/wlasl2000_filtered_keypoints"

# Google Drive file ID
FILE_ID = "1l8pguR-nfbo5kLPua9wirHVPqiG1BE5h"

# Direct download URL
URL = f"https://drive.google.com/uc?id={FILE_ID}"

ZIP_PATH = "wlasl2000_filtered_keypoints.zip"

# Download only if dataset doesn't exist
if not os.path.exists(DATASET_PATH):
    print("Downloading dataset...")

    gdown.download(URL, ZIP_PATH, quiet=False)

    print("Extracting dataset...")

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_ref:
        zip_ref.extractall("assets/text_to_sign")

    os.remove(ZIP_PATH)

    print("Dataset ready ✅")

else:
    print("Dataset already exists ✅")
    