# pulling the Fashion MNIST dataset into my workspace

import os
import urllib
import urllib.request
from zipfile import ZipFile

URL = "https://nnfs.io/datasets/fashion_mnist_images.zip"
FILE = "fashion_mnist_images.zip"
FOLDER = "fashion_mnist_images"

if not os.path.isfile(FILE):
    print(f"Downloading {URL} and saving as {FILE}")
    urllib.request.urlretrieve(url=URL, filename=FILE)

print("Unzipping image files...")
with ZipFile(file=FILE) as zip_images:
    zip_images.extractall(path=FOLDER)
    
print("Completed")