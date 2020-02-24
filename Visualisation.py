import json
import matplotlib.pyplot as plt
import numpy as np

def plot_learning_rate():
    with open("./learning_rate.json","r") as dump_f:
        load_dict = json.load(dump_f)


    for i,(k,v) in enumerate(load_dict.items()):
        print((k,v))
    keys = list(load_dict.keys())

    markers = ',v^8sph'
    i = 0
    for key in keys:
        plt.plot(load_dict[key]['EPOCH'],load_dict[key]['val_loss_list'],marker = markers[i])
        i += 1

    #设置坐标轴名称
    plt.xlabel('Epochs')
    plt.ylabel('Logloss')
    #设置坐标轴刻度
    my_x_ticks = np.arange(0, len(load_dict[key]['EPOCH']), 1)
    plt.xticks(my_x_ticks)

    plt.legend(keys, loc='upper left')
    plt.show()


def plot_one_learning_rate():
    with open("./learning_rate.json","r") as dump_f:
        load_dict = json.load(dump_f)


    for i,(k,v) in enumerate(load_dict.items()):
        print((k,v))
    keys = list(load_dict.keys())

    key = keys[0]
    epoch = load_dict[key]['EPOCH']
    train_loss = load_dict[key]['train_loss_list']
    val_loss = load_dict[key]['val_loss_list']
    val_auc = [i for i in load_dict[key]['val_auc_list']]
    val_acc = [i/100 for i in load_dict[key]['val_acc_list']]


    plt.plot(epoch,train_loss)
    plt.plot(epoch, val_loss)
    plt.plot(epoch, val_auc)
    plt.plot(epoch, val_acc)

    plt.legend(['train_loss', 'val_loss','val_auc','val_acc'], loc='upper left')
    plt.show()

plot_learning_rate()
