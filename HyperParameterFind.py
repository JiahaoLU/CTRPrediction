from Training_Validation_Test import *
import json


class HyperFinder():
    def __init__(self):
        self.DATASET_PATH = "./Data/train.csv"
        self.EPOCH = 10
        self.LEARNING_RATE = [pow(10,-1-i/4) for i in range(1,9)]
        self.BATCH_SIZE = [40*i for i in range(1,11)]
        self.WEIGHT_DECAY = [i*1e-6 for i in range(1,11)]
        self.EMBED_DIM = 10
        self.save_best = True

        self.best_learning_rate = None
        self.model_best = None
        self.epoch_list = []
        self.train_loss_list = []
        self.val_acc_list = []
        self.val_acc_list = []
        self.val_loss_list = []
        self.auc_best = 1
        self.acc_best = 0
        self.loss_best = 999999
        self.Total_Epoch = 1
        self.data_save = {}

    def save_model(self,trained_model,*args):
        model_name = ''
        for i in args:
            model_name += (i + ' ')
        torch.save(trained_model, f'{model_name}.pt')

    def obj_2_json(self):
        load_dict = self.data_save
        with open("./save.json","w") as dump_f:
            json.dump(load_dict,dump_f)

    def find_best_Learning_Rate(self):
        BATCH_SIZE = 12800
        WEIGHT_DECAY = 1e-6
        for LEARNING_RATE in self.LEARNING_RATE:
            self.data_save[str(LEARNING_RATE)] = {}
            self.data_save[str(LEARNING_RATE)]['EPOCH'] = []
            self.data_save[str(LEARNING_RATE)]['train_loss_list'] = []
            self.data_save[str(LEARNING_RATE)]['val_auc_list'] = []
            self.data_save[str(LEARNING_RATE)]['val_acc_list'] = []
            self.data_save[str(LEARNING_RATE)]['val_loss_list'] = []
            self.data_save[str(LEARNING_RATE)]['test_auc'] = 999999
            self.data_save[str(LEARNING_RATE)]['test_acc'] = 0
            self.data_save[str(LEARNING_RATE)]['test_loss'] = 999999
            trained_model,epoch_list,train_loss_list,val_auc_list,val_acc_list,val_loss_list,test_auc,test_acc,test_loss = \
                main_process(self.DATASET_PATH, self.EPOCH, LEARNING_RATE, BATCH_SIZE, WEIGHT_DECAY, self.EMBED_DIM,60*10)
            self.data_save[str(LEARNING_RATE)]['EPOCH'] = epoch_list
            self.data_save[str(LEARNING_RATE)]['train_loss_list'] = train_loss_list
            self.data_save[str(LEARNING_RATE)]['val_auc_list'] = val_auc_list
            self.data_save[str(LEARNING_RATE)]['val_acc_list'] = val_acc_list
            self.data_save[str(LEARNING_RATE)]['val_loss_list'] = val_loss_list
            self.data_save[str(LEARNING_RATE)]['val_loss_list'] = test_auc
            self.data_save[str(LEARNING_RATE)]['val_loss_list'] = test_acc
            self.data_save[str(LEARNING_RATE)]['test_loss'] = test_loss
            if test_loss < self.loss_best:
                self.best_learning_rate = LEARNING_RATE
                self.epoch_list = epoch_list
                self.auc_best = test_auc
                self.acc_best = test_acc
                self.loss_best = test_loss
                self.Total_Epoch = len(epoch_list)
                self.model_best = trained_model

        if self.save_best:
            self.save_model()


def main():
    hyperFinder = HyperFinder()
    hyperFinder.find_best_Learning_Rate()
    hyperFinder.obj_2_json()


if __name__ == "__main__":
    main()

