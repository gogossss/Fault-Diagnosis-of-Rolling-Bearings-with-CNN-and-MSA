# github：https://github.com/boating-in-autumn-rain?tab=repositories
# 微信公众号：秋雨行舟
# B站：秋雨行舟
#
# 该项目涉及数据集以及相关安装包在公众号《秋雨行舟》回复轴承即可领取。
# 对于该项目有疑问的可以在B站留言（免费答疑），或者联系微信（有偿）：LettersLive23
# 该项目对应的视频可在B站搜索《秋雨行舟》进行观看学习。
# 欢迎交流学习，共同进步


from time import sleep
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, ShuffleSplit
from OriginalVibrationSignal import ovs_preprocess
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import random
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from datetime import datetime
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE


# 如果是GPU，需要去掉注释，如果是CPU，则注释
# gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
# assert len(gpu) == 1
# tf.config.experimental.set_memory_growth(gpu[0], True)

def subtime(date1, date2):
    return date2 - date1


num_classes = 10  # 样本类别
length = 784  # 样本长度
number = 300  # 每类样本的数量
normal = True  # 是否标准化
rate = [0.5, 0.25, 0.25]  # 测试集验证集划分比例

path = r'data/0HP'
x_train, y_train, x_valid, y_valid, x_test, y_test = ovs_preprocess.prepro(
    d_path=path,
    length=length,
    number=number,
    normal=normal,
    rate=rate,
    enc=False, enc_step=28)

x_train = np.array(x_train)
y_train = np.array(y_train)
x_valid = np.array(x_valid)
y_valid = np.array(y_valid)
x_test = np.array(x_test)
y_test = np.array(y_test)

print(x_train.shape)
print(x_valid.shape)
print(x_test.shape)
print(y_train.shape)
print(y_valid.shape)
print(y_test.shape)

y_train = [int(i) for i in y_train]
y_valid = [int(i) for i in y_valid]
y_test = [int(i) for i in y_test]

# 打乱顺序
index = [i for i in range(len(x_train))]
random.seed(1)
random.shuffle(index)
x_train = np.array(x_train)[index]
y_train = np.array(y_train)[index]

index1 = [i for i in range(len(x_valid))]
random.shuffle(index1)
x_valid = np.array(x_valid)[index1]
y_valid = np.array(y_valid)[index1]

index2 = [i for i in range(len(x_test))]
random.shuffle(index2)
x_test = np.array(x_test)[index2]
y_test = np.array(y_test)[index2]

print(x_train.shape)
print(x_valid.shape)
print(x_test.shape)
print(y_train)
print(y_valid)
print(y_test)
print("x_train的最大值和最小值：", x_train.max(), x_train.min())
print("x_test的最大值和最小值：", x_test.max(), x_test.min())

x_train = tf.reshape(x_train, (len(x_train), 784, 1))
x_valid = tf.reshape(x_valid, (len(x_valid), 784, 1))
x_test = tf.reshape(x_test, (len(x_test), 784, 1))


# 保存最佳模型
class CustomModelCheckpoint(keras.callbacks.Callback):
    def __init__(self, model, path):
        self.model = model
        self.path = path
        self.best_loss = np.inf

    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs['val_loss']
        if val_loss < self.best_loss:
            print("\nValidation loss decreased from {} to {}, saving model".format(self.best_loss, val_loss))
            self.model.save_weights(self.path, overwrite=True)
            self.best_loss = val_loss


# t-sne初始可视化函数
def start_tsne():
    print("正在进行初始输入数据的可视化...")
    x_train1 = tf.reshape(x_train, (len(x_train), 784))
    X_tsne = TSNE().fit_transform(x_train1)
    plt.figure(figsize=(10, 10))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_train)
    plt.colorbar()
    plt.show()


# start_tsne()
# sleep(600000)
len_x_train = len(x_train)
len_x_test = len(x_test)

x_train = tf.reshape(x_train, (len_x_train, 784))
x_test = tf.reshape(x_test, (len_x_test, 784))

x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)

# ======================================================================
# 模型训练
rfc_model = RandomForestClassifier(n_jobs=-1)
cv_split = ShuffleSplit(n_splits=5)
score_ndarray = cross_val_score(rfc_model, x_train, y_train, cv=cv_split)
print(score_ndarray.round(4))
print(score_ndarray.mean().round(4))

# 模型测试
random_tree_model = RandomForestClassifier(n_jobs=-1)
random_tree_model.fit(x_train, y_train)
print("测试集准确率为： ", random_tree_model.score(x_test, y_test).round(4))
