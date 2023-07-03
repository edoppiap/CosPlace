import os
import gdown
import shutil

URLS = {
    "tokyo_xs": "https://drive.google.com/file/d/1lztMekUYPi4gXNzC6I04NHa-Z-twgxCs/view?usp=share_link",
    "sf_xs": "https://drive.google.com/file/d/1gcPnxOJQqpwbs0cO8TkGFRu_-NIEdOry/view?usp=sharing",
}

os.makedirs("data", exist_ok=True)
for dataset_name, url in URLS.items():
    print(f"Downloading {dataset_name}")
    zip_filepath = f"data/{dataset_name}.zip"
    gdown.download(url, zip_filepath, fuzzy=True)
    shutil.unpack_archive(zip_filepath, extract_dir="data")
    os.remove(zip_filepath)
