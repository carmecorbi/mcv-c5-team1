import os
from pathlib import Path

# Define the paths of the two folders
folder1 = Path("GlobalWheat/train/images")  # Change to your actual folder path
folder2 = Path("GlobalWheat/train/labels")  # Change to your actual folder path

# Get the set of filenames (without extensions) in each folder
files1 = {f.stem for f in folder1.iterdir() if f.is_file()}  # Filenames without extensions
files2 = {f.stem for f in folder2.iterdir() if f.is_file()}  # Filenames without extensions

# Find files that exist in folder1 but not in folder2
missing_files = files1 - files2

# Print missing file names
if missing_files:
    print("Files in the first folder but NOT in the second:")
    for file in sorted(missing_files):
        print(file)
else:
    print("All files in the first folder exist in the second folder.")