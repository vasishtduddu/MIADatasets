import argparse
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import numpy as np
import wget
import tarfile

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision import datasets
from collections import defaultdict
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision.transforms as transforms


def create_dir_if_doesnt_exist(path_to_dir: Path) -> Optional[Path]:

    resolved_path: Path = path_to_dir.resolve()

    if not resolved_path.exists():
        print("{} does not exist. Creating...")

        try:
            resolved_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print("Error occurred while creating directory {}. Exception: {}".format(resolved_path, e))
            return None

        print("{} created.".format(resolved_path))
    else:
        print("{} already exists.".format(resolved_path))

    return resolved_path


def configure(raw_path):
    raw_path: Path = Path(raw_path)
    purchase_path: Path = raw_path / "Purchase100"
    create_dir_if_doesnt_exist(purchase_path)
    downloaded_purchase: Path = purchase_path / "dataset_purchase.tgz"
    wget.download("https://raw.githubusercontent.com/vasishtduddu/MIADatasets/main/dataset_purchase.tgz", str(downloaded_purchase))
    tar = tarfile.open(downloaded_purchase)
    tar.extractall(purchase_path)
    tar.close()
    (purchase_path / "dataset_purchase").replace(purchase_path / "purchase")


def tensor_data_create(features, labels):

    tensor_x = torch.stack([torch.FloatTensor(i) for i in features]) # transform to torch tensors
    tensor_y = torch.stack([torch.LongTensor([i]) for i in labels])[:,0]
    dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y)
    return dataset

def prepare_purchase_data(datapath):

    print("Loading Purchase Data")
    DATASET_PATH = datapath / "Purchase100" / "purchase"
    data_set =np.genfromtxt(DATASET_PATH,delimiter=',')

    X = data_set[:,1:].astype(np.float64)
    Y = (data_set[:,0]).astype(np.int32)-1

    train_data = X[:19732]
    test_data = X[19732:19732+19732]

    train_label = Y[:19732]
    test_label = Y[19732:19732+19732]

    train_dataset = tensor_data_create(train_data, train_label)
    test_dataset = tensor_data_create(test_data, test_label)

    return train_dataset, test_dataset


def test(model, loader, args):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(args.device), target.to(args.device)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum().item()

        accuracy = 100. * correct / len(loader.sampler)
        print("Accuracy: {}/{} ({}%)".format(correct,len(loader.sampler),accuracy))
    return accuracy

def train(epochs, model, trainloader, testloader, optimizer, args):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(trainloader):
            data, target = data.to(args.device), target.to(args.device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            #loss = nn.NLLLoss(output,target)
            loss.backward()
            optimizer.step()

            if batch_idx % args.batch_size == 0:
                done = batch_idx * len(data)
                percentage = 100. * batch_idx / len(trainloader.sampler)
                print(f'Train Epoch: {epoch} Loss: {loss.item():.6f}')

        test(model, trainloader, args)
        test(model, testloader, args)
    return model

def test_overlap(model, trainloader, testloader, args):
    model.eval()
    with torch.no_grad():
        for ((traindata, traintarget), (testdata, testtarget)) in zip(trainloader, testloader):
            traindata, traintarget = traindata.to(args.device), traintarget.to(args.device)
            output_train = model(traindata)
            pred_train = output_train.data.max(1, keepdim=True)[1]

            testdata, testtarget = testdata.to(args.device), testtarget.to(args.device)
            output_test = model(testdata)
            pred_test = output_test.data.max(1, keepdim=True)[1]



class PurchaseClassifier(nn.Module):
    def __init__(self,num_classes=100):
        super(PurchaseClassifier, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(600,1024),
            nn.Tanh(),
            nn.Linear(1024,512),
            nn.Tanh(),
            nn.Linear(512,256),
            nn.Tanh(),
            nn.Linear(256,128),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(128,num_classes)


    def forward(self,x):
        hidden_out = self.features(x)
        return self.classifier(hidden_out)


def main(args: argparse.Namespace) -> None:

    raw_path: Path = Path(args.raw_path)
    maybe_results_path: Optional[Path] = create_dir_if_doesnt_exist(raw_path)

    num_classes = 100
    train_dataset, test_dataset = prepare_purchase_data(raw_path)
    model = PurchaseClassifier(num_classes=num_classes).to(args.device)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if args.load == "False":

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

        train(args.epochs, model, trainloader, testloader, optimizer, args)
        if args.save:
            model_path = "purchase.pkl"
            torch.save(model.state_dict(), model_path)
            print("Model Saved!!")

    else:
        model = PurchaseClassifier(num_classes=num_classes).to(args.device)
        model_path = "purchase.pkl"
        model.load_state_dict(torch.load(model_path))
        model.eval()
        # evaluation of model on members
        test(model, trainloader, args)
        # evaluation of model on non-members
        test(model, testloader, args)
        # overlap between train and test data loader
        test_overlap(model, trainloader, testloader, args)

def handle_args() -> argparse.Namespace:

    parser: argparse.ArgumentParser = argparse.ArgumentParser()

    parser.add_argument("--raw_path", type = str, default = "./data/", help = "Location to store raw datasets.")
    parser.add_argument("--lr", type = float, default = 1e-3, help = "Learning Rate")
    parser.add_argument("--decay", type = float, default = 0, help = "Weight decay/L2 Regularization")
    parser.add_argument("--batch_size", type = int, default = 128, help = "Batch size for training data")
    parser.add_argument("--device", type = str, default = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), help = "GPU/CPU")
    parser.add_argument("--save", type = bool, default = True, help = "Save model")
    parser.add_argument("--epochs", type = int, default = 50, help = "Number of Model Training Iterations")
    parser.add_argument("--load", type = str, default = "False", help = "Load trained model")
    args: argparse.Namespace = parser.parse_args()
    return args


if __name__ == "__main__":

    args = handle_args()
    if (Path(args.raw_path) / "Purchase100").exists():
        msg: str = "Datasets exist."
        print(msg)
    else:
        msg: str = "Configuring datasets."
        print(msg)
        configure(args.raw_path)
    main(args)
