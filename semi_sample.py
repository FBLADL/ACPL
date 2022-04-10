import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer(classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14])
Labels = {
    "No Finding": 14,
    "Atelectasis": 0,
    "Cardiomegaly": 1,
    "Effusion": 2,
    "Infiltration": 3,
    "Mass": 4,
    "Nodule": 5,
    "Pneumonia": 6,
    "Pneumothorax": 7,
    "Consolidation": 8,
    "Edema": 9,
    "Emphysema": 10,
    "Fibrosis": 11,
    "Pleural_Thickening": 12,
    "Hernia": 13,
}
gr = pd.read_csv("Data_Entry_2017.csv", index_col=0)
gr = gr.to_dict()["Finding Labels"]
img_path = "train_val_list.txt"
with open(img_path) as f:
    names = f.read().splitlines()
imgs = np.asarray([x for x in names])
gr = np.asarray([gr[i] for i in imgs])
binary_gr = np.zeros((gr.shape[0], 15))
for idx, i in enumerate(gr):
    target = i.split("|")
    binary_result = mlb.fit_transform([[Labels[i] for i in target]]).squeeze()
    binary_gr[idx] = binary_result


# count label percentage
# count_idx = list()
# count_idx_sampled = list()
# selected_imgs = list()
for time in range(3):
    count_idx = list()
    count_idx_sampled = list()
    selected_imgs = list()
    for i in range(15):
        temp_count = np.nonzero(binary_gr[:, i])[0]
        np.random.shuffle(temp_count)
        count_idx.append(temp_count)
        temp_count_sampled = temp_count[: int(temp_count.shape[0] * 0.02)]
        count_idx_sampled.append(temp_count_sampled)
        selected_imgs.append(imgs[temp_count_sampled].tolist())
    selected_imgs = set(sum(selected_imgs, []))
    with open("train_list_2_{}.txt".format(time + 1), "w") as f:
        for i in selected_imgs:
            f.write(i)
            f.write("\n")
# 20%
