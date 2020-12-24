import os

PATH = 'D:/assets/data/FER_dataset(emotions)/data/'

train_count = 0
test_count = 0

classes = os.listdir(os.path.join(PATH, "test/"))

for cl in classes:
    print(cl, ":", len(os.listdir(os.path.join(PATH, "train", cl))))
    train_count += len(os.listdir(os.path.join(PATH, "train", cl)))

print(train_count)

for cl in classes:
    print(cl, ":", len(os.listdir(os.path.join(PATH, "test", cl))))
    test_count += len(os.listdir(os.path.join(PATH, "test", cl)))

print(test_count)