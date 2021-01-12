import keras
from keras.datasets import mnist
from keras.preprocessing import image
from keras.applications import imagenet_utils
import numpy as np
import random
import os
import nn_util


def load_mnist_tarin_data(include_classes=[]):
    (x_train, y_train), (_, _) = get_mnist_data()
    train_imgs = []
    train_labels = []
    train_imgids = []
    for x_idx in range(0, len(x_train)):
        if np.argmax(y_train[x_idx]) in include_classes:
            train_imgs.append(x_train[x_idx])
            train_labels.append(y_train[x_idx])
            train_imgids.append(x_idx)
    return np.array(train_imgs), np.array(train_labels), np.array(train_imgids)


def load_mnist_test_data(include_classes=[]):
    (_, _), (x_test, y_test) = get_mnist_data()
    test_imgs = []
    test_labels = []
    test_imgids = []
    for x_idx in range(0, len(x_test)):
        if np.argmax(y_test[x_idx]) in include_classes:
            test_imgs.append(x_test[x_idx])
            test_labels.append(y_test[x_idx])
            test_imgids.append(x_idx)
    return np.array(test_imgs), np.array(test_labels), np.array(test_imgids)


def get_mnist_data(img_rows=28, img_cols=28, channels=1, is_standardization=True):
    input_shape = (img_rows, img_cols, channels)
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # 处理 x
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    if is_standardization:
        x_train /= 255
        x_test /= 255
    # 处理 y
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)
    return (x_train, y_train), (x_test, y_test)


def get_imagenet_val_classname(val_file='./inputs/imagenet/ILSVRC2012_val_label.txt'):
    id_to_labelname = {}
    with open(val_file, encoding='UTF-8') as f:
        for line in f.readlines():
            els = line.strip().split(" ")
            id_to_labelname[els[0]] = els[1]
    return id_to_labelname


def load_imgnet_val_data(include_classes=[], imgn_path="./inputs/imagenet/ILSVRC2012_img_val"):
    id_to_classname = get_imagenet_val_classname()
    inputShape = (224, 224)
    imgs = []
    labels = []
    img_ids = []
    num_2_n, n_2_num, full_info = get_class_mapping()
    for img_index in range(1, 50001):
        img_id = "%08d" % (img_index)
        img_file_prefix = "ILSVRC2012_val_" + img_id
        img_class = n_2_num[id_to_classname[img_file_prefix]]
        if img_class not in include_classes:
            continue
        img_path = imgn_path + "/ILSVRC2012_val_" + str(img_id) + ".JPEG"
        # img_paths.append(img_path)
        img = image.load_img(img_path, target_size=(inputShape[0], inputShape[1]))
        img = image.img_to_array(img)
        imgs.append(img)
        labels.append(nn_util.get_target_onehot_label(img_class, 1000))
        img_ids.append(img_index)

    imgs = np.array(imgs)
    imgs = imagenet_utils.preprocess_input(imgs)
    return imgs, labels, img_ids


def sample_imgnet_test_data(st_idx=0, end_idx=100, imgn_path="./inputs/imagenet/ILSVRC2012_img_test"):
    inputShape = (224, 224)
    random.seed(1)
    arr = random.sample(range(1, 100000 + 1), 100000)
    arr = arr[st_idx:end_idx]

    imgs = []
    for img_index in range(0, end_idx - st_idx):
        img_id = "%08d" % (arr[img_index])
        img_path = imgn_path + "/ILSVRC2012_test_" + str(img_id) + ".JPEG"
        # img_paths.append(img_path)
        img = image.load_img(img_path, target_size=(inputShape[0], inputShape[1]))
        img = image.img_to_array(img)
        imgs.append(img)
    imgs = np.array(imgs)
    imgs = imagenet_utils.preprocess(imgs)
    return imgs


def get_class_mapping(mapping_file='./inputs/imagenet/ILSVRC2012_mapping.txt'):
    num_2_n = {}
    n_2_num = {}
    full_info = []
    with open(mapping_file, encoding='UTF-8') as f:
        for line in f.readlines():
            els = line.strip().split(" ")
            els[0] = int(els[0])
            num_2_n[els[0]] = els[1]
            n_2_num[els[1]] = els[0]
            full_info.append(els)
    return num_2_n, n_2_num, full_info


def load_imagenet_train_data(include_class=[], dir='./inputs/imagenet/ILSVRC2012_img_train'):
    list_file = os.listdir(dir)
    inputShape = (224, 224)
    imgs = []
    labels = []
    preprocess = imagenet_utils.preprocess_input
    _, n_2_num, _ = get_class_mapping()
    class_num = len(n_2_num)
    img_ids=[]
    for file in list_file:
        if file.endswith('JPEG'):
            img_path = os.path.join(dir, file)
            img_infos = file.split('_')
            img_class = img_infos[0]
            img_class = n_2_num[img_class]

            if img_class not in include_class:
                continue

            label = nn_util.get_target_onehot_label(img_class, class_num)
            img = image.load_img(img_path, target_size=(inputShape[0], inputShape[1]))
            img = image.img_to_array(img)
            imgs.append(img)
            labels.append(label)
            img_id = int(img_infos[1].split('.')[0])
            img_ids.append(img_id)
    imgs = preprocess(np.array(imgs))
    return imgs, labels, img_ids


def check_model(model):
    # define a dictionary that maps model names to their classes inside Keras
    MODELS = ('vgg19', 'resnet50', 'lenet1', 'lenet4', 'lenet5')
    # esnure a valid model name was supplied via command line argument
    if model not in MODELS:
        raise AssertionError("The model should in the: " + str(MODELS))


def preprocess_imagenet_image(img_path, model):
    check_model(model)
    inputShape = (224, 224)
    preprocess = imagenet_utils.preprocess_input
    if model in ("inception", "xception"):
        inputShape = (299, 299)
        preprocess = keras.applications.inception_v3.preprocess_input
    img = image.load_img(img_path, target_size=inputShape)
    input_img_data = image.img_to_array(img)
    input_img_data = np.expand_dims(input_img_data, axis=0)
    input_img_data = preprocess(input_img_data)
    return input_img_data


def preprocess_mnist_image(img_path):
    img = image.load_img(img_path, target_size=(28, 28))
    input_img_data = image.img_to_array(img)
    input_img_data = input_img_data[:, :, 0].reshape(28, 28, 1)
    input_img_data = np.expand_dims(input_img_data, axis=0)
    input_img_data /= 255.0
    return input_img_data


def deprocess_imagenet_image(x, model):
    check_model(model)
    if model in ('vgg', 'resnet', 'vgg19', 'resnet50'):
        x = x.reshape((224, 224, 3))
        # Remove zero-center by mean pixel
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        # 'BGR'->'RGB'
        x = x[:, :, ::-1]

    elif model in ('inception', 'xception'):
        x = x.reshape((299, 299, 3))
        x /= 2.
        x += 0.5
        x *= 255.
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def deprocess_mnist_image(x):
    x *= 255.0
    x = np.clip(x, 0, 255).astype('uint8')
    return x.reshape(x.shape[0], x.shape[1])


# def get_all_imagenet_tests_labels():
#     from keras.applications.resnet50 import ResNet50
#     labels = {}
#     each_label_imgs = {}
#     imgn_path = "./inputs/ILSVRC2012_img_test"
#     inputShape = (224, 224)
#     preprocess = imagenet_utils.preprocess_input
#     total_imgs = 100000
#     model = ResNet50(weights='imagenet')
#     for img_idx in range(1, total_imgs + 1):
#         img_id = "%08d" % (img_idx)
#         img_path = imgn_path + "/ILSVRC2012_test_" + str(img_id) + ".JPEG"
#         img = image.load_img(img_path, target_size=(inputShape[0], inputShape[1]))
#         img = image.img_to_array(img)
#         img = np.expand_dims(img, axis=0)
#         img = preprocess(img)
#         pred = np.argmax(model.predict(img)[0])
#         labels[img_id] = pred
#         if pred not in each_label_imgs:
#             each_label_imgs[pred] = []
#         each_label_imgs[pred].append(img_id)
#         print("predict {0}, and the prediction is {1}.".format(img_path, pred))
#     return labels, each_label_imgs


# if __name__ == '__main__':
#     sample_imgnet_val_data(1, 101)
    #     labels, each_label_imgs = get_all_imagenet_tests_labels()
    #     print(labels, each_label_imgs)
    #     labels_file = "./outputs/imagenet/labels.pkl"
    #     each_label_imgs_file = "./outputs/imagenet/each_label_imgs.pkl"
    #     IOUtils.write_pkl_to_file(labels, labels_file, 'wb')
    #     IOUtils.write_pkl_to_file(each_label_imgs, each_label_imgs_file, 'wb')
    # get_class_mapping()
