import os
import medmnist
from medmnist import INFO

# Create a test directory
test_dir = os.path.join('test_montage')
os.makedirs(test_dir, exist_ok=True)

# Choose a dataset
data_flag = 'pathmnist'
info = INFO[data_flag]
DataClass = getattr(medmnist, info['python_class'])

# Load the dataset with size=64
train_dataset = DataClass(split='train', download=True, size=64)

# Create the montage
preview_dir = os.path.join(test_dir, 'preview')
os.makedirs(preview_dir, exist_ok=True)

print(f"Creating montage for {data_flag}...")
frames = train_dataset.montage(length=10, save_folder=preview_dir)
print(f"Saved montage to {preview_dir}")

print("Done!") 