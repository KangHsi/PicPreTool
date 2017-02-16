
import os
import pickle
from PIL import Image
import numpy as np
import sklearn
import sklearn.linear_model
import matplotlib.pyplot as plt


def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='iso-8859-1')
    fo.close()
    return dict


# 调用sklearn对数据进行shuffle操作
def shuffle_data(data, labels):
    data, _, labels, _ = sklearn.cross_validation.train_test_split(
        data, labels, test_size=0.0, random_state=42
    )


    return data, labels


def load_data(train_file):
    d = unpickle(train_file)
    data = d['data']
    fine_labels = d['fine_labels']
    length = len(d['fine_labels'])

    data, labels = shuffle_data(
        data,
        np.array(fine_labels)
    )
    return (
        data.reshape(length, 3, 32, 32),
        labels
    )


if __name__ == '__main__':
    cifar_python_directory = os.path.abspath('cifar-100-python')

    print('Converting...')
    cifar_caffe_directory = os.path.abspath('cifar100_train_lmdb')
if not os.path.exists(cifar_caffe_directory):
    # X, y_f = load_data(os.path.join(cifar_python_directory, 'train'))
    Xt, yt_f = load_data(os.path.join(cifar_python_directory, 'train'))


print('Data is fully loaded,now truly convertung.')

# 图像样本显示
for index in range(Xt.shape[0]):
    a = Xt[index]
    label=yt_f[index]
    dir=str(cifar_python_directory  + "\\"+str(label) +"\\")

    r = Image.fromarray(a[0]).convert('L')
    g = Image.fromarray(a[1]).convert('L')
    b = Image.fromarray(a[2]).convert('L')
    image = Image.merge("RGB", (r, g, b))
    # plt.imshow(image)
    # plt.show()
    png = str(dir + str(index) + ".png")
    if not os.path.exists(dir):
        os.makedirs(dir)

        if not os.path.exists(png):
            image.save(png, 'png')
    else:
        if not os.path.exists(png):
            image.save(png, 'png')


#