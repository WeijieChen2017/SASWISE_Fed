# please load balanced_siim_4fold.json and create a dataloader for the test set

import json
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    NormalizeIntensityd,
    CenterSpatialCropd,
    ToTensord,
    EnsureTyped)

from monai.data import CacheDataset

import time


fold_index = 0
cache_rate = 1.0
num_workers = 32

with open('balanced_siim_4fold.json', 'r') as f:
    data = json.load(f)

test_index = [fold_index % 4]
train_index = [fold_index+1 % 4, fold_index+2 % 4]
val_index = [fold_index+3 % 4]

test_data = data[f'fold_{test_index[0]}']
train_data = data[f'fold_{train_index[0]}'] + data[f'fold_{train_index[1]}']
val_data = data[f'fold_{val_index[0]}']

# print the first 5 items in each of the data
print(f"The test dataset is fold_{fold_index}, with {len(test_data)} items")
print(f"The train dataset is fold_{fold_index+1} and {fold_index+2}, with {len(train_data)} items")
print(f"The val dataset is fold_{fold_index+3}, with {len(val_data)} items")
print("-"*30)
print(f"The first 3 items in the test dataset are:")
for item in test_data[:3]:
    print(item)

print(f"The first 3 items in the train dataset are:")
for item in train_data[:3]:
    print(item)

print(f"The first 3 items in the val dataset are:")
for item in val_data[:3]:
    print(item) 

# now create a dataloader function
def create_dataloader(data, batch_size=1, shuffle=False, num_workers=8):
    dataset = CacheDataset(
        data=data,
        transform=Compose([
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]), 
            NormalizeIntensityd(keys=["image"]),
            CenterSpatialCropd(keys=["image", "label"], roi_size=(128,128,32)),
            ToTensord(keys=["image", "label"]),
            EnsureTyped(keys=["image", "label"])
        ]),
        cache_rate=cache_rate,
        num_workers=num_workers,   
    )

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        persistent_workers=True, 
        pin_memory=True, 
        prefetch_factor=4, 
        drop_last=True
    )
    return dataloader

# now create a dataloader for the train, val and test sets
# train_dataloader = create_dataloader(train_data, batch_size=1, shuffle=True, num_workers=num_workers)
# val_dataloader = create_dataloader(val_data, batch_size=1, shuffle=False, num_workers=num_workers)

# Now record the time it takes to load the data for each dataloader at first 5 batches
# for name, dataloader in [("test", test_dataloader), ("train", train_dataloader), ("val", val_dataloader)]:

to_analyze = ["test"]

for name in to_analyze:
    print(f"Current testing {name} dataloader")
    start_time = time.time()
    if name == "test":
        dataloader = create_dataloader(test_data, batch_size=1, shuffle=False, num_workers=num_workers)
    end_time = time.time()
    print(f"Time taken to create {name} dataloader: {end_time - start_time} seconds")
    total_time = 0
    for i in range(5):
        start_time = time.time()
        for batch in dataloader:
            pass
        end_time = time.time()
        print(f"Batch {i+1} time: {end_time - start_time} seconds")
        start_time = end_time
        total_time += end_time - start_time
    print(f"Total time for {name} dataloader: {total_time} seconds at cache rate {cache_rate} and {num_workers} workers")
    print("-"*30)
