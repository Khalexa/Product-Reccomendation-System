import subprocess
import pathlib

raw_dir = pathlib.Path("data/raw")
raw_dir.mkdir(parents=True, exist_ok=True)

kaggle_path = r"C:\Users\H\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.13_qbz5n2kfra8p0\LocalCache\local-packages\Python313\Scripts\kaggle.exe"

subprocess.run([
    kaggle_path,
    "datasets", "download",
    "-d", "retailrocket/ecommerce-dataset",
    "--unzip",
    "-p", str(raw_dir)
])
print(f"Dataset downloaded and extracted to {raw_dir}")
