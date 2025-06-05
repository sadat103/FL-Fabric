import argparse
import os
import json
import itertools
from datasets import load_dataset
from PIL import ImageEnhance, Image
import gc

num_clients = 20

output_dir = "federated_skin_2024"
ds_train = load_dataset("Pranavkpba2000/skin_cancer_dataset", split="train")  # still arrow‑backed, but not list()
ds_test = load_dataset("Pranavkpba2000/skin_cancer_dataset", split="test")  # still arrow‑backed, but not list()

def preprocess_image(image):
    image = ImageEnhance.Sharpness(image).enhance(2.0)
    image = ImageEnhance.Contrast(image).enhance(1.5)
    image = ImageEnhance.Brightness(image).enhance(1.2)
    return image

for i in range(num_clients):
    shard_train = ds_train.shard(num_shards=num_clients, index=i)
    shard_test = ds_test.shard(num_shards=num_clients, index=i)
    client_dir = os.path.join(output_dir, f"client_{i+1}")
    
    client_dir_train = os.path.join(client_dir, "train")
    os.makedirs(client_dir_train, exist_ok=True)

    client_dir_test = os.path.join(client_dir, "test")
    os.makedirs(client_dir_test, exist_ok=True)
    
    
    labels_train = []
    labels_test = []
    for j, example in enumerate(shard_train):
        img = preprocess_image(example["image"])
        fname = f"{j}.jpg"
        img.save(os.path.join(client_dir_train, fname))
        labels_train.append({"filename": fname, "label": example["label"]})

    for j, example in enumerate(shard_test):
        img = preprocess_image(example["image"])
        fname = f"{j}.jpg"
        img.save(os.path.join(client_dir_test, fname))
        labels_test.append({"filename": fname, "label": example["label"]})
    
    with open(os.path.join(client_dir_train, "labels.json"), "w") as f:
        json.dump(labels_train, f)
    with open(os.path.join(client_dir_test, "labels.json"), "w") as f:
        json.dump(labels_test, f)