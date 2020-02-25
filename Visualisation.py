import json
import matplotlib.pyplot as plt
import numpy as np
from numpy import log10


def plot_learning_rate():
    with open("./learning_rate.json", "r") as dump_f:
        load_dict = json.load(dump_f)

    for i, (k, v) in enumerate(load_dict.items()):
        print((k, v))
    keys = list(load_dict.keys())
    keys.reverse()

    markers = 'v^8sph'
    i = 0
    for key in keys:
        plt.plot(load_dict[key]['EPOCH'], [log10(i) for i in load_dict[key]['val_loss_list']], marker=markers[i])
        i += 1

    plt.xlabel('Epochs')
    plt.ylabel('log(Logloss)')

    my_x_ticks = np.arange(0, len(load_dict[keys[0]]['EPOCH']), 5)
    plt.xticks(my_x_ticks)

    plt.legend(keys, loc='upper right')
    # plt.figure()
    plt.savefig("lr_plot.png", dpi=300)
    plt.show()


def plot_weight_decay():
    with open("./Weight_decay.json", "r") as dump_f:
        load_dict = json.load(dump_f)

    for i, (k, v) in enumerate(load_dict.items()):
        print((k, v))
    keys = list(load_dict.keys())
    keys.reverse()

    markers = 'v^8sph1234Dd'
    i = 0
    for key in keys:
        plt.plot(load_dict[key]['EPOCH'], [log10(i) for i in load_dict[key]['val_loss_list']], marker=markers[i])
        i += 1

    plt.xlabel('Epochs')
    plt.ylabel('log(Logloss)')

    my_x_ticks = np.arange(0, len(load_dict[keys[0]]['EPOCH']), 5)
    plt.xticks(my_x_ticks)

    plt.legend(keys, loc='upper right')
    # plt.figure()
    plt.savefig("wd_plot.png", dpi=300)
    plt.show()


def plot_embed_dim():
    with open("./EMBED_DIM.json", "r") as dump_f:
        load_dict = json.load(dump_f)

    for i, (k, v) in enumerate(load_dict.items()):
        print((k, v))
    keys = list(load_dict.keys())
    keys.reverse()

    markers = 'v^8sph1234Dd'
    i = 0
    for key in keys:
        plt.plot(load_dict[key]['EPOCH'], [log10(i) for i in load_dict[key]['val_loss_list']], marker=markers[i])
        i += 1

    plt.xlabel('Epochs')
    plt.ylabel('log(Logloss)')

    my_x_ticks = np.arange(0, len(load_dict[keys[0]]['EPOCH']), 5)
    plt.xticks(my_x_ticks)

    plt.legend(keys, loc='upper right')
    # plt.figure()
    plt.savefig("ed_plot.png", dpi=300)
    plt.show()


def plot_one_learning_rate():
    with open("./learning_rate.json", "r") as dump_f:
        load_dict = json.load(dump_f)

    for i, (k, v) in enumerate(load_dict.items()):
        print((k, v))
    keys = list(load_dict.keys())

    key = keys[0]
    epoch = load_dict[key]['EPOCH']
    train_loss = load_dict[key]['train_loss_list']
    val_loss = load_dict[key]['val_loss_list']
    val_auc = [i for i in load_dict[key]['val_auc_list']]
    val_acc = [i / 100 for i in load_dict[key]['val_acc_list']]

    plt.plot(epoch, train_loss)
    plt.plot(epoch, val_loss)
    plt.plot(epoch, val_auc)
    plt.plot(epoch, val_acc)

    plt.legend(['train_loss', 'val_loss', 'val_auc', 'val_acc'], loc='upper left')
    plt.show()


plot_learning_rate()
plot_weight_decay()
plot_embed_dim()