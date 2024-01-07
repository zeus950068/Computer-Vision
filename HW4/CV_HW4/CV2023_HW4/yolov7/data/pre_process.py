import os
import glob
import random
import argparse

CLASS_NAME = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# Q2 and Q3 TODO : select "better" images from Q2 folder
def select_imaegs(image_paths, images_num=200):
    """
    :param image_paths: --> ['your_folder/images1.jpg', 'your_folder/images2.jpg', ...]
    :param images_num: choose the number of images
    :return :
        selected_image_paths = ['your_folder/images10.jpg', 'your_folder/images12.jpg', ...]
    """
    # TODO : select images
    # WAY1
    # random.shuffle(image_paths)
    # selected_image_paths = image_paths[:images_num]

    # WAY2
    selected_image_paths = []
    folder_paths = {}
    for path in image_paths:
      folder = os.path.dirname(path)
      if folder not in folder_paths:
          folder_paths[folder] = []
      folder_paths[folder].append(path)
    print(folder_paths)
    for key, values in folder_paths.items():
      values_list = [values] if isinstance(values, str) else values  # 確保 values 是列表
      num_to_select = min(33, len(values))
      selected_values = random.sample(values_list, num_to_select)
      selected_image_paths.extend(selected_values)

    selected_keys = random.sample(folder_paths.keys(), 2)
    for key in selected_keys:
      values = folder_paths[key]
      selected_value_last = random.choice(values)
      # print(selected_value_last)
      selected_image_paths.append(selected_value_last)

    # WAY3
    # selected_image_paths = []
    # selected_paths_170 = [path for path in image_paths if '/CityCam/Q3/170' in path]
    # random.shuffle(selected_paths_170)
    # selected_image_paths_temp = selected_paths_170[:24]
    # selected_image_paths = selected_image_paths + selected_image_paths_temp

    # selected_paths_173 = [path for path in image_paths if '/CityCam/Q3/173' in path]
    # random.shuffle(selected_paths_173)
    # selected_image_paths_temp = selected_paths_173[:35]
    # selected_image_paths = selected_image_paths + selected_image_paths_temp

    # selected_paths_398 = [path for path in image_paths if '/CityCam/Q3/398' in path]
    # random.shuffle(selected_paths_398)
    # selected_image_paths_temp = selected_paths_398[:39]
    # selected_image_paths = selected_image_paths + selected_image_paths_temp

    # selected_paths_410 = [path for path in image_paths if '/CityCam/Q3/410' in path]
    # random.shuffle(selected_paths_410)
    # selected_image_paths_temp = selected_paths_410[:31]
    # selected_image_paths = selected_image_paths + selected_image_paths_temp

    # selected_paths_495 = [path for path in image_paths if '/CityCam/Q3/495' in path]
    # random.shuffle(selected_paths_495)
    # selected_image_paths_temp = selected_paths_495[:44]
    # selected_image_paths = selected_image_paths + selected_image_paths_temp

    # selected_paths_511 = [path for path in image_paths if '/CityCam/Q3/511' in path]
    # random.shuffle(selected_paths_511)
    # selected_image_paths_temp = selected_paths_511[:27]
    # selected_image_paths = selected_image_paths + selected_image_paths_temp
    # print(selected_image_paths)

    ###################################Q3###########################################
    with open('/content/drive/MyDrive/Colab Notebooks/CV_HW4/Q2_txt/train.txt', 'r') as file1:
        paths1 = file1.readlines()
    with open('/content/drive/MyDrive/Colab Notebooks/CV_HW4/Q2_txt/val.txt', 'r') as file2:
        paths2 = file2.readlines()

    Q2_select = paths1 + paths2
    selected_image_paths.extend(Q2_select)
    # print(selected_image_paths)



    #####################################################################################################################################
    from collections import Counter
    parent_folders = [os.path.dirname(path) for path in selected_image_paths]   # 使用 os.path.dirname 獲取每個路徑的父文件夾
    for path in selected_image_paths:
      folder_counts = Counter(parent_folders)   # 統計各個文件夾的數量

    if 'Q2' in path:
      print("NOW IS IN Q2")
      folder_170_count = folder_counts['../CityCam/Q2/170']
      print(f"在 170 資料夾的路徑數量：{folder_170_count}")
      folder_173_count = folder_counts['../CityCam/Q2/173']
      print(f"在 173 資料夾的路徑數量：{folder_173_count}")
      folder_398_count = folder_counts['../CityCam/Q2/398']
      print(f"在 398 資料夾的路徑數量：{folder_398_count}")
      folder_410_count = folder_counts['../CityCam/Q2/410']
      print(f"在 410 資料夾的路徑數量：{folder_410_count}")
      folder_495_count = folder_counts['../CityCam/Q2/495']
      print(f"在 495 資料夾的路徑數量：{folder_495_count}")
      folder_511_count = folder_counts['../CityCam/Q2/511']
      print(f"在 511 資料夾的路徑數量：{folder_511_count}")

    elif 'Q3' in path:
      print("NOW IS IN Q3")
      folder_170_count = folder_counts['../CityCam/Q3/170']
      print(f"在 170 資料夾的路徑數量：{folder_170_count}")
      folder_173_count = folder_counts['../CityCam/Q3/173']
      print(f"在 173 資料夾的路徑數量：{folder_173_count}")
      folder_398_count = folder_counts['../CityCam/Q3/398']
      print(f"在 398 資料夾的路徑數量：{folder_398_count}")
      folder_410_count = folder_counts['../CityCam/Q3/410']
      print(f"在 410 資料夾的路徑數量：{folder_410_count}")
      folder_495_count = folder_counts['../CityCam/Q3/495']
      print(f"在 495 資料夾的路徑數量：{folder_495_count}")
      folder_511_count = folder_counts['../CityCam/Q3/511']
      print(f"在 511 資料夾的路徑數量：{folder_511_count}")

    return selected_image_paths

# TODO : split train and val images
def split_train_val_path(all_image_paths, train_val_ratio=0.9):
    """
    :param all_image_paths: all image paths for question in the data folder
    :param train_val_ratio: ratio of image paths used to split training and validation
    :return :
        train_image_paths = ['your_folder/images1.jpg', 'your_folder/images2.jpg', ...]
        val_image_paths = ['your_folder/images3.jpg', 'your_folder/images4.jpg', ...]
    """
    # TODO : split train and val
    #WAY1
    # train_image_paths = all_image_paths[: int(len(all_image_paths) * train_val_ratio)]  # just an example
    # val_image_paths = all_image_paths[int(len(all_image_paths) * train_val_ratio):]  # just an example
    #########################################################################################################################
    #WAY2
    import numpy as np
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt

    rng = np.random.default_rng()

    num_samples = len(all_image_paths)
    num_train = int(train_val_ratio * num_samples)

    indices = np.arange(num_samples)
    rng.shuffle(indices)

    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    
    train_image_paths = [all_image_paths[i] for i in train_indices]
    val_image_paths = [all_image_paths[i] for i in val_indices]
    #########################################################################################################################
    # WAY3
    from sklearn.model_selection import train_test_split
    # train_image_paths, val_image_paths = train_test_split(all_image_paths, 
    #                             random_state=777, 
    #                             train_size=train_val_ratio, 
    #                             shuffle=True)

    return train_image_paths, val_image_paths


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default='./data/CityCam', help='path of CityCam datasets folder')
    parser.add_argument('--ques', type=str, default='Q1', choices=['Q1', 'Q2', 'Q3'], help='question in data_folder')
    args = parser.parse_args()
    print(args)

    # Get whole and Test image paths
    all_image_paths = glob.glob(os.path.join(args.data_folder, args.ques, '*', '*.jpg'))
    test_image_paths = glob.glob(os.path.join(args.data_folder, 'test', '*' + os.sep + '*.jpg'))

    # for Q2 and Q3 : select images
    if args.ques == 'Q2' or args.ques == 'Q3':
        selected_image_paths = select_imaegs(all_image_paths, images_num=200)
    else:
        selected_image_paths = all_image_paths
    # split Train and Val
    train_image_paths, val_image_paths = split_train_val_path(selected_image_paths)
    

    # write train/val/test info
    train_path = os.path.join(args.data_folder, 'train.txt')
    val_path = os.path.join(args.data_folder, 'val.txt')
    test_path = os.path.join(args.data_folder, 'test.txt')
    count_train = 0   # add myself
    with open(train_path, 'w') as f:
        for image_path in train_image_paths:
            f.write(os.path.abspath(image_path) + '\n')
            count_train += 1   # add myself

    count_val = 0   # add myself
    with open(val_path, 'w') as f:
        for image_path in val_image_paths:
            f.write(os.path.abspath(image_path) + '\n')
            count_val += 1   # add myself

    count_test = 0   # add myself
    with open(test_path, 'w') as f:
        for image_path in test_image_paths:
            f.write(os.path.abspath(image_path) + '\n')
            count_test += 1   # add myself

    print(f"Number of paths written to train.txt: {count_train}")   # add myself
    print(f"Number of paths written to val.txt: {count_val}")   # add myself
    print(f"Number of paths written to test.txt: {count_test}")   # add myself


    # write training YAML file
    with open('./data/citycam.yaml', 'w') as f:
        f.write("train: " + os.path.abspath(train_path) + "\n")
        f.write("val: " + os.path.abspath(val_path) + "\n")
        f.write("test: " + os.path.abspath(test_path) + "\n")
        # number of classes
        f.write('nc: 80\n')
        # class names
        f.write('names: ' + str(CLASS_NAME))

    # delete cache
    if os.path.exists(os.path.join(args.data_folder, 'train.cache')):
        os.remove(os.path.join(args.data_folder, 'train.cache'))
    if os.path.exists(os.path.join(args.data_folder, 'val.cache')):
        os.remove(os.path.join(args.data_folder, 'val.cache'))
    """
    if os.path.exists(os.path.join(args.data_folder, 'test.cache')):
        os.remove(os.path.join(args.data_folder, 'test.cache'))
      """
