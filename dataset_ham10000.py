from datasets import load_dataset
import os
import json
import numpy as np
from collections import defaultdict
from PIL import Image
from PIL import ImageEnhance

from datasets import load_dataset
import os, json
from PIL import Image

num_clients = 20
output_dir = "federated_skin_cancer"
ds_train = load_dataset("marmal88/skin_cancer", split="train")  # arrow dataset
ds_test = load_dataset("marmal88/skin_cancer", split="test")  # arrow dataset

def preprocess_image(image):
    image = ImageEnhance.Sharpness(image).enhance(2.0)
    image = ImageEnhance.Contrast(image).enhance(1.5)
    image = ImageEnhance.Brightness(image).enhance(1.2)
    return image


for i in range(num_clients):
    client    = f"client_{i+1}"
    client_dir = os.path.join(output_dir, client)
    os.makedirs(client_dir, exist_ok=True)
    
    client_dir_train = os.path.join(client_dir, "train")
    os.makedirs(client_dir_train, exist_ok=True)

    client_dir_test = os.path.join(client_dir, "test")
    os.makedirs(client_dir_test, exist_ok=True)

    # 1/20th of the data for this client
    shard_train = ds_train.shard(num_shards=num_clients, index=i)
    shard_test = ds_test.shard(num_shards=num_clients, index=i)

    labels_train = []
    labels_test = []
    for j, example in enumerate(shard_train):
        img = preprocess_image(example["image"])
        fname = f"{j}.jpg"
        img.save(os.path.join(client_dir_train, fname))
        labels_train.append({"filename": fname, "label": example["dx"]})

    for j, example in enumerate(shard_test):
        img = preprocess_image(example["image"])
        fname = f"{j}.jpg"
        img.save(os.path.join(client_dir_test, fname))
        labels_test.append({"filename": fname, "label": example["dx"]})

    # dump client labels once
    with open(os.path.join(client_dir_train, "labels.json"), "w") as f:
        json.dump(labels_train, f)
    with open(os.path.join(client_dir_test, "labels.json"), "w") as f:
        json.dump(labels_test, f)

print("✅ Sharded IID split complete.")


# # Load the dataset
# dataset = load_dataset("marmal88/skin_cancer")

# # Print dataset sample to check the structure
# print(dataset['train'][0])

# # Number of FL clients
# num_clients = 20

# # Split dataset into IID (Balanced Data Distribution)
# def split_iid(dataset, num_clients):
#     data = list(dataset['train'])
#     np.random.shuffle(data)
#     client_data = np.array_split(data, num_clients)
#     return {f'client_{i+1}': list(client_data[i]) for i in range(num_clients)}

# # Split dataset into Non-IID (Grouped by Cancer Type)
# def split_non_iid(dataset, num_clients):
#     data_by_class = defaultdict(list)

#     # Group data by 'dx' (diagnosis type)
#     for item in dataset['train']:
#         label = item['dx']  # Get the class label (e.g., 'melanoma')
#         data_by_class[label].append(item)

#     # Distribute classes to different clients
#     client_data = {f'client_{i+1}': [] for i in range(num_clients)}
#     class_keys = list(data_by_class.keys())

#     for i, class_key in enumerate(class_keys):
#         client_data[f'client_{(i+1) % num_clients}'].extend(data_by_class[class_key])

#     return client_data

# # Choose IID or Non-IID split
# federated_data = split_non_iid(dataset, num_clients)

# # Create folder for federated dataset
# output_dir = "federated_skin_cancer"
# os.makedirs(output_dir, exist_ok=True)

# # Save federated data for each client
# for client, data in federated_data.items():
#     client_dir = os.path.join(output_dir, client)
#     os.makedirs(client_dir, exist_ok=True)

#     # Save images & labels
#     labels = []
#     for i, item in enumerate(data):
#         # Save image
#         image = item['image']
#         image_path = os.path.join(client_dir, f"{i}.jpg")
#         image.save(image_path)  # Save as JPEG

#         # Store label
#         labels.append({"filename": f"{i}.jpg", "label": item["dx"]})

#     # Save label metadata as JSON
#     with open(os.path.join(client_dir, "labels.json"), "w") as f:
#         json.dump(labels, f)

# print("✅ Federated dataset created successfully!")