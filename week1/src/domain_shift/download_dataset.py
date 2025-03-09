import kagglehub

# Download latest version
path = kagglehub.dataset_download("slavkoprytula/aquarium-data-cots")

print("Path to dataset files:", path)