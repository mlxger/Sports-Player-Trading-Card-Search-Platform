import os
import shutil

root_images = "G:/PROJECT/图片裁剪/project/code/yolo_images/"
root_labels = "G:/PROJECT/图片裁剪/project/code/yolo_labels/"

labels = os.listdir(root_labels)

trains = labels[:int(len(labels) * 0.7)]
tests_tmp = labels[int(len(labels) * 0.7):]
tests = tests_tmp[:int(len(tests_tmp) * 0.8)]
vals = tests_tmp[int(len(tests_tmp) * 0.8):]

# 创建文件夹
# train
train_path = "G:/PROJECT/图片裁剪/project/code/embImgs/train/"
train_path_image = "G:/PROJECT/图片裁剪/project/code/embImgs/train/images"
train_path_label = "G:/PROJECT/图片裁剪/project/code/embImgs/train/labels"
os.makedirs(train_path, exist_ok=True)
os.makedirs(train_path_image, exist_ok=True)
os.makedirs(train_path_label, exist_ok=True)

# test
test_path = "G:/PROJECT/图片裁剪/project/code/embImgs/test/"
test_path_image = "G:/PROJECT/图片裁剪/project/code/embImgs/test/images"
test_path_label = "G:/PROJECT/图片裁剪/project/code/embImgs/test/labels"
os.makedirs(test_path, exist_ok=True)
os.makedirs(test_path_image, exist_ok=True)
os.makedirs(test_path_label, exist_ok=True)

# val
val_path = "G:/PROJECT/图片裁剪/project/code/embImgs/val/"
val_path_image = "G:/PROJECT/图片裁剪/project/code/embImgs/val/images"
val_path_label = "G:/PROJECT/图片裁剪/project/code/embImgs/val/labels"
os.makedirs(val_path, exist_ok=True)
os.makedirs(val_path_image, exist_ok=True)
os.makedirs(val_path_label, exist_ok=True)

for t in trains:
    index = t.split(".")[0]
    image_path = root_images + index + ".jpg"
    if not os.path.exists(image_path):
        continue
    shutil.move(image_path, train_path_image)
    label_path = root_labels + index + ".txt"
    shutil.move(label_path, train_path_label)


for t in tests:
    index = t.split(".")[0]
    image_path = root_images + index + ".jpg"
    if not os.path.exists(image_path):
        continue
    shutil.move(image_path, test_path_image)
    label_path = root_labels + index + ".txt"
    shutil.move(label_path, test_path_label)


for t in vals:
    index = t.split(".")[0]
    image_path = root_images + index + ".jpg"
    if not os.path.exists(image_path):
        continue
    shutil.move(image_path, val_path_image)
    label_path = root_labels + index + ".txt"
    shutil.move(label_path, val_path_label)


if __name__ == '__main__':
    pass
