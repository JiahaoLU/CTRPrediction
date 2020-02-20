import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from Data_Preprocessor import *
from Deep_Model import FieldAwareFactorizationMachineModel


def run_train(model, optimizer, data_loader, criterion, device, log_interval=10):
    """
    train the model using backward propagation
    :param model: the model to be trained. instance of subclass of nn.Module
    :param optimizer: torch optimiser with learning rate
    :param data_loader: torch DataLoader of training data set
    :param criterion: nn.BCELoss
    :param device: CUDA GPU or CPU
    :param log_interval: interval for showing training loss
    :return: none
    """
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


def run_test(model, data_loader, device):
    """
    evaluate / test the model
    :param model: the model to be evaluated/tested. instance of subclass of nn.Module
    :param data_loader: torch DataLoader of eval/test data set
    :param device: CUDA GPU or CPU
    :return: auc score, accuracy of prediction
    """
    model.eval()
    targets, predicts = list(), list()
    correct = 0
    with torch.no_grad():
        for fields, target in data_loader:
            fields, target = fields.to(device), target.to(device)
            y = model(fields)
            targets.extend(target.tolist())
            predicts.extend(y.tolist())
            predict_click = torch.round(y.data)
            correct += (predict_click == target).sum().item()
    return roc_auc_score(targets, predicts), correct / len(targets) * 100


def main_process(dataset_path, epoch, learning_rate, batch_size, weight_decay, embeddim):
    """
    Main process for train/evaluate/test the model, determine the hyper parameters here.
    :param dataset_path: path of the original csv file
    :param epoch: number of epochs
    :param learning_rate: learning rate of gradient descent
    :param batch_size: size of batches
    :param weight_decay: L2 regularisation
    :param embeddim: dimension of latent vector
    :return: the trained model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Prepare the data
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

    # Prepare the model and loss function
    model = FieldAwareFactorizationMachineModel(field_dims, embed_dim=embeddim).to(device)
    criterion = torch.nn.BCELoss()  # binary cross entropy loss
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Training
    for epoch_i in range(epoch):
        run_train(model, optimizer, train_data_loader, criterion, device)
        auc, acc = run_test(model, valid_data_loader, device)
        print('epoch:', epoch_i, 'validation: auc:', auc, '--- acc:', acc)

    # Test
    auc, acc = run_test(model, test_data_loader, device)
    print('test auc:', auc, 'test acc:', acc)

    return model


def main(save_model=False):
    """
    Main function. Set hyper parameters here. Determine whether save the model or not.
    :param save_model: boolean to determine if save the model
    :return:
    """
    DATASET_PATH = "./Data/train20k.csv"
    EPOCH = 10
    LEARNING_RATE = 0.001
    BATCH_SIZE = 200
    WEIGHT_DECAY = 1e-6
    EMBED_DIM = 4
    trained_model = main_process(DATASET_PATH, EPOCH, LEARNING_RATE, BATCH_SIZE, WEIGHT_DECAY, EMBED_DIM)

    if save_model:
        model_name = "FFM"
        torch.save(trained_model, f'{model_name}.pt')


if __name__ == "__main__":
    main()
