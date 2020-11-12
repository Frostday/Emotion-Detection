import pandas as pd
import os
import shutil

ORIGINAL_PATH = 'D:/assets/data/emotions/images/'
NEW_PATH = 'data/'

cols = ['id', 'image', 'emotion']
df = pd.read_csv("D:/assets/data/emotions/data/500_picts_satz.csv", names=cols)
df.drop('id', inplace=True, axis=1)

for i, x in df.iterrows():
    # print(x['image'], x['emotion'])
    if x['emotion']=='happiness':
        img_path = os.path.join(ORIGINAL_PATH, x['image'])
        destination_path = os.path.join(NEW_PATH, "happy")
        shutil.copy(img_path, destination_path)
    if x['emotion']=='fear':
        img_path = os.path.join(ORIGINAL_PATH, x['image'])
        destination_path = os.path.join(NEW_PATH, "scared")
        shutil.copy(img_path, destination_path)
    if x['emotion']=='neutral':
        img_path = os.path.join(ORIGINAL_PATH, x['image'])
        destination_path = os.path.join(NEW_PATH, "neutral")
        shutil.copy(img_path, destination_path)
    if x['emotion']=='anger':
        img_path = os.path.join(ORIGINAL_PATH, x['image'])
        destination_path = os.path.join(NEW_PATH, "angry")
        shutil.copy(img_path, destination_path)
    if x['emotion']=='sad':
        img_path = os.path.join(ORIGINAL_PATH, x['image'])
        destination_path = os.path.join(NEW_PATH, "sad")
        shutil.copy(img_path, destination_path)

print("All done!")