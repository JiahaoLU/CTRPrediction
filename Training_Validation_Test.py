import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from Data_Preprocessor import *
from Deep_Model import FieldAwareFactorizationMachineModel
from torchvision import datasets, transforms, models
import torch.optim as optim
from sklearn.model_selection import train_test_split


def run_train(model, optimizer, data_loader, criterion, device, log_interval=100):
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
        if (i + 1) % log_interval == 0:
            print('    - loss:', total_loss / log_interval)
            total_loss = 0


def run_test(model, data_loader, device):
    model.eval()
    targets, predicts = list(), list()
    with torch.no_grad():
        for fields, target in data_loader:
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
    return roc_auc_score(targets, predicts)


def main_process(dataset_path, epoch, learning_rate, batch_size, weight_decay):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = DataPreprocessor(dataset_path)
    train_length = int(len(dataset) * 0.8)
    valid_length = int(len(dataset) * 0.1)
    test_length = len(dataset) - train_length - valid_length
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, (train_length, valid_length, test_length))
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size)
    valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size)
    field_dims = dataset.get_field_dims()

    model = FieldAwareFactorizationMachineModel(field_dims, embed_dim=4).to(device)
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch_i in range(epoch):
        run_train(model, optimizer, train_data_loader, criterion, device)
        auc = run_test(model, valid_data_loader, device)
        print('epoch:', epoch_i, 'validation: auc:', auc)
    auc = run_test(model, test_data_loader, device)
    print('test auc:', auc)
    return model


def main(save_model=False):
    DATASET_PATH = "./Data/smaller_train.csv"
    EPOCH = 10
    LEARNING_RATE = 0.001
    BATCH_SIZE = 200
    WEIGHT_DECAY = 1e-6
    trained_model = main_process(DATASET_PATH, EPOCH, LEARNING_RATE, BATCH_SIZE, WEIGHT_DECAY)

    if save_model:
        model_name = "FFM"
        torch.save(trained_model, f'{model_name}.pt')


if __name__ == "__main__":
    main()
