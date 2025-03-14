import kagglehub

# Download latest version
path = kagglehub.dataset_download("usmanafzaal/strawberry-disease-detection-dataset")

print("Path to dataset files:", path)