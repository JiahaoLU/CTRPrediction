import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from Data_Preprocessor import *
from Deep_Model import FieldAwareFactorizationMachineModel
from torchvision import datasets, transforms, models
import torch.optim as optim
from sklearn.model_selection import train_test_split
import threading


def fun_timer():
    global boom
    boom = True

def run_train(model, optimizer, data_loader, criterion, device, log_interval=1):
    model.train()
    total_loss = 0
    for i, (fields, target) in enumerate(data_loader):
        fields, target = fields.to(device), target.to(device)
        y = model(fields)
        loss = criterion(y, target.float())
        model.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if i % log_interval == 0:
            print('    - loss:', total_loss / log_interval)
            total_loss = 0
    return loss.item()


def run_test(model, data_loader, device, criterion):
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
    model.eval()
    targets, predicts = list(), list()
    correct = 0
    with torch.no_grad():
        for fields, target in data_loader:
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            loss = criterion(y, target.float())
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
            predict_click = torch.round(y.data)
            correct += (predict_click == target).sum().item()
    return roc_auc_score(targets, predicts), correct / len(targets) * 100, loss.item()



def main_process(dataset_path, epoch, learning_rate, batch_size, weight_decay, embeddim,boomtime = 60):
    # time boom for adjusting hyperparameters to control the same training time cost
    global boom
    boom = False
    timer = threading.Timer(boomtime, fun_timer)
    timer.start()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())

    dataset = DataPreprocessor(dataset_path)
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length))
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size,shuffle=True,num_workers=4,pin_memory=True)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size,shuffle=True,num_workers=4,pin_memory=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size,shuffle=True,num_workers=4,pin_memory=True)
    field_dims = dataset.get_field_dims()

    model = FieldAwareFactorizationMachineModel(field_dims, embed_dim=embeddim).to(device)
    criterion = torch.nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    epoch_list = []
    train_loss_list = []
    val_auc_list = []
    val_acc_list = []
    val_loss_list = []
    for epoch_i in range(epoch):
        train_loss = run_train(model, optimizer, train_data_loader, criterion, device)
        val_auc, val_acc,val_loss = run_test(model, valid_data_loader, device, criterion)
        epoch_list.append(epoch_i)
        train_loss_list.append(train_loss)
        val_auc_list.append(val_auc)
        val_acc_list.append(val_acc)
        val_loss_list.append(val_loss)
        print('epoch:', epoch_i, 'train loss:', train_loss, 'validation: auc:', val_auc, '--- acc:', val_acc,'--- loss:', val_loss)
        if boom:
            print('time up, break')
            break
        else:
            print(timer)
    test_auc, test_acc, test_loss = run_test(model, test_data_loader, device, criterion)
    print('test auc:', test_auc, 'test acc:', test_acc, 'test loss', test_loss)
    return model,epoch_list,train_loss_list,val_auc_list,val_acc_list,val_loss_list,test_auc,test_acc,test_loss


def main(save_model=True):
    DATASET_PATH = "./Data/train20k.csv"
    EPOCH = 1000
    LEARNING_RATE = 0.001
    BATCH_SIZE = 3200
    WEIGHT_DECAY = 1e-6
    EMBED_DIM = 10
    trained_model,epoch_list,train_loss_list,val_auc_list,val_acc_list,val_loss_list,auc,acc,loss = \
        main_process(DATASET_PATH, EPOCH, LEARNING_RATE, BATCH_SIZE, WEIGHT_DECAY, EMBED_DIM)

    if save_model:
        model_name = "FFM"
        torch.save(trained_model, f'{model_name}.pt')



if __name__ == "__main__":
    main()
