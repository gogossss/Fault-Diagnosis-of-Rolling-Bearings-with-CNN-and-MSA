from time import sleep
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random
from datetime import datetime
import numpy as np
from sklearn.manifold import TSNE
import tensorflow as tf
import matplotlib.pyplot as plt
import xlwt
import pandas as pd

def write_excel_xls(path, sheet_name, value):
    writer = pd.ExcelWriter(path)		# 写入Excel文件
    value.to_excel(writer, sheet_name, float_format='%.5f')		# ‘page_1’是写入excel的sheet名
    writer._save()
    print('写入excel完成')

    writer.close()
    

def plot_data1(x_train, path):
    t = np.arange(0, len(x_train[0]), 1)
    f, ax = plt.subplots(5, 2, figsize=(10, 8))
    f.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.3, wspace=0.3)

    # f.suptitle('Original Signals')
    ax[0][0].set_title('12k_Drive_End_B007_0_118')
    ax[0][0].plot(t, list(x_train[0]), 'b', linewidth=0.2)
    # ax[0][0].set_ylabel(' Value of Data ')
    
    ax[1][0].set_title('12k_Drive_End_B014_0_185')
    ax[1][0].plot(t, list(x_train[75]), 'b', linewidth=0.2)
    # ax[1][0].set_ylabel(' Value of Data ')
    
    ax[2][0].set_title('12k_Drive_End_B021_0_222')
    ax[2][0].plot(t, list(x_train[150]), 'b', linewidth=0.2)
    ax[2][0].set_ylabel(' Value of Data ', fontsize=20)

    ax[3][0].set_title('12k_Drive_End_IR007_0_105')
    ax[3][0].plot(t, list(x_train[225]), 'b', linewidth=0.2)
    # ax[3][0].set_ylabel(' Value of Data ')
    
    ax[4][0].set_title('12k_Drive_End_IR014_0_169')
    ax[4][0].plot(t, list(x_train[300]), 'b', linewidth=0.2)
    ax[4][0].set_xlabel(' Sampling point ', fontsize=20)
    # ax[4][0].set_ylabel(' Value of Data ', fontsize=16)
    
    ax[0][1].set_title('12k_Drive_End_IR021_0_209')
    ax[0][1].plot(t, list(x_train[375]), 'b', linewidth=0.2)

    ax[1][1].set_title('12k_Drive_End_OR007@6_0_130')
    ax[1][1].plot(t, list(x_train[450]), 'b', linewidth=0.2)
    
    ax[2][1].set_title('12k_Drive_End_OR014@6_0_197')
    ax[2][1].plot(t, list(x_train[525]), 'b', linewidth=0.2)
    # ax[2][1].set_ylabel(' Value of Data ', fontsize=20)

    ax[3][1].set_title('12k_Drive_End_OR021@6_0_234')
    ax[3][1].plot(t, list(x_train[600]), 'b', linewidth=0.2)
    
    ax[4][1].set_title('normal_0_97')
    ax[4][1].plot(t, list(x_train[675]), 'b', linewidth=0.2)
    ax[4][1].set_xlabel(' Sampling point ', fontsize=20)
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
    plt.close()

def plot_data(x_train, path):
    t = np.arange(0, len(x_train[0]), 1)
    f, ax = plt.subplots(5, 1, figsize=(10, 8))
    f.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.3, wspace=0.3)

    # f.suptitle('Original Signals')
    ax[0].set_title('B_A_1')
    ax[0].plot(t, list(x_train[0]), 'b', linewidth=1)
    # ax[0][0].set_ylabel(' Value of Data ')
    
    ax[1].set_title('C_A_1')
    ax[1].plot(t, list(x_train[1]), 'b', linewidth=1)
    # ax[1][0].set_ylabel(' Value of Data ')
    
    ax[2].set_title('H_A_1')
    ax[2].plot(t, list(x_train[2]), 'b', linewidth=1)
    ax[2].set_ylabel(' Value of Data ', fontsize=20)

    ax[3].set_title('I_A_1')
    ax[3].plot(t, list(x_train[3]), 'b', linewidth=1)
    # ax[3][0].set_ylabel(' Value of Data ')
    
    ax[4].set_title('O_A_1')
    ax[4].plot(t, list(x_train[4]), 'b', linewidth=1)
    ax[4].set_xlabel(' Sampling point ', fontsize=20)
    # ax[4][0].set_ylabel(' Value of Data ', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(path)
    plt.show()
    plt.close()

# 输入数据可视化
def start_tsne(x_train, y_train, path, per=30):
    print("正在进行初始输入数据的可视化...")
    print(len(x_train))
    x_train1 = tf.reshape(x_train, (len(x_train), len(x_train[0])))
    X_tsne = TSNE(perplexity=per, n_components=2).fit_transform(x_train1)
    plt.figure(figsize=(10, 10))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_train)
    plt.colorbar(ticks=np.linspace(0, int(len(x_train)/75)-1, int(len(x_train)/75)))
    plt.savefig(path)
    plt.show()
    plt.close()


# 绘制混淆矩阵
def confusion(y_train, y_test, y_pred_int, path):
    con_mat = confusion_matrix(y_test.astype(str), y_pred_int.astype(str))
    print(con_mat)
    classes = list(set(y_train))
    classes.sort()
    plt.imshow(con_mat, cmap=plt.cm.Blues)
    indices = range(len(con_mat))
    plt.xticks(indices, classes)
    plt.yticks(indices, classes)
    plt.colorbar()
    plt.xlabel('Predict')
    plt.ylabel('True')
    for first_index in range(len(con_mat)):
        for second_index in range(len(con_mat[first_index])):
            plt.text(first_index, second_index, con_mat[second_index][first_index], va='center', ha='center')
    plt.savefig(path)
    plt.show()
    plt.close()
    

    

