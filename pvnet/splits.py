import os, random, csv

root = "../datasets/LINEMOD/cat/JPEGImages"  # adjust to your images folder
image_ids = [f.split(".")[0] for f in os.listdir(root) if f.endswith(".png")]

random.shuffle(image_ids)
split = int(0.8 * len(image_ids))

train_ids = image_ids[:split]
test_ids = image_ids[split:]

with open("../datasets/LINEMOD/cat/train_split.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows([[id] for id in train_ids])

with open("../datasets/LINEMOD/cat/test_split.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows([[id] for id in test_ids])
