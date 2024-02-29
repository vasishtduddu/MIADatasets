import argparse
import os
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import numpy as np
import wget
import shutil
import tarfile
import pickle
import pandas as pd
import torch.utils.data as Data
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split


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
    texas_path: Path = raw_path / "Texas100X"
    create_dir_if_doesnt_exist(texas_path)
    downloaded_texas: Path = texas_path / "dataset_texas.tgz"
    wget.download("https://github.com/vasishtduddu/MIADatasets/raw/main/dataset_texas_sattr.tar.gz", str(downloaded_texas))

    tar = tarfile.open(downloaded_texas)
    tar.extractall(texas_path)
    tar.close()
    (texas_path / "dataset_texas_sattr" /  "texas_100_v2_feature_desc.p").replace(texas_path / "texas_100_v2_feature_desc.p")
    (texas_path / "dataset_texas_sattr" / "texas_100_v2_features.p").replace(texas_path / "texas_100_v2_features.p")
    (texas_path / "dataset_texas_sattr" / "texas_100_v2_labels.p").replace(texas_path / "texas_100_v2_labels.p")
    shutil.rmtree(texas_path / "dataset_texas_sattr")
    return texas_path


def tensor_data_create(features, labels):

    tensor_x = torch.stack([torch.FloatTensor(i) for i in features]) # transform to torch tensors
    tensor_y = torch.stack([torch.LongTensor([i]) for i in labels])[:,0]
    dataset = torch.utils.data.TensorDataset(tensor_x,tensor_y)
    return dataset


def process_features(features, attribute_dict, max_attr_vals, target_attr, col_flags, skip_sensitive=False, skip_corr=True):
    """
    Returns the feature matrix after expanding the nominal features, 
    and removing the features that are not needed for model training.
    """
    features = pd.DataFrame(features)
    # for removing sensitive feature
    if skip_sensitive:
        features.drop(columns=target_attr, inplace=True)
    if col_flags['TEXAS_THCIC_ID'] != None:
        features.drop(columns=col_flags['TEXAS_THCIC_ID'], inplace=True)
    # skip_corr flag is used to decide whether to skip the correlated attribute in Texas-100X
    if skip_corr == True:
        if col_flags['TEXAS_ETHN'] in target_attr:
            features.drop(columns=col_flags['TEXAS_RACE'], inplace=True)
    # for expanding categorical features
    if attribute_dict != None:
        for col in attribute_dict:
            # skip in case the sensitive feature was removed above
            if col not in features.columns:
                continue
            # to expand the non-binary categorical features
            if max_attr_vals[col] != 1:
                features[col] *= max_attr_vals[col]
                features[col] = pd.Categorical(features[col], categories=range(int(max_attr_vals[col])+1))
        features = pd.get_dummies(features)
    Z_race = features.loc[:,[col_flags['TEXAS_ETHN']]]
    Z_sex = features.loc[:,[1]]
    Z = pd.concat([Z_race,Z_sex], axis=1)
    return np.array(features), Z

def load_data(datapath, args):
    """
    Loads the training and test sets for the given data set.
    """
    DATASET_LABELS = datapath / "texas_100_v2_labels.p"
    DATASET_FEATURES = datapath / "texas_100_v2_features.p"
    DATASET_FEATURES_SENSITIVE = datapath / "texas_100_v2_feature_desc.p"

    X = np.load(DATASET_FEATURES,allow_pickle=True)
    Y = np.load(DATASET_LABELS,allow_pickle=True)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.5, random_state=42)
    X_train, X_test, y_train, y_test = X_train[:10000], X_test[:10000], y_train[:10000], y_test[:10000]

    target_attrs, attribute_dict, max_attr_vals, col_flags = get_sensitive_features(X_train, DATASET_FEATURES_SENSITIVE, size=3)
    X_train, Z_train = process_features(X_train, attribute_dict, max_attr_vals, target_attrs, col_flags)
    target_attrs, attribute_dict, max_attr_vals, col_flags = get_sensitive_features(X_test, DATASET_FEATURES_SENSITIVE, size=3)
    X_test, Z_test = process_features(X_test, attribute_dict, max_attr_vals, target_attrs, col_flags)

    # train_dataset = tensor_data_create(X_train, y_train)
    # test_dataset = tensor_data_create(X_test, y_test)

    train_dataset = Data.TensorDataset(torch.from_numpy(np.array(X_train)).type(torch.FloatTensor), torch.from_numpy(np.array(y_train)).type(torch.LongTensor), torch.from_numpy(np.array(Z_train)).type(torch.LongTensor))
    test_dataset = Data.TensorDataset(torch.from_numpy(np.array(X_test)).type(torch.FloatTensor), torch.from_numpy(np.array(y_test)).type(torch.LongTensor), torch.from_numpy(np.array(Z_test)).type(torch.LongTensor))

    return train_dataset, test_dataset


def get_sensitive_features(data, datapath_sensfeat, size=3):
    """
    Returns the sensitive features for a given data set, along 
    with the attribute dictionary if available.
    """
    col_flags = {x: None for x in ['TEXAS_RACE', 'TEXAS_ETHN', 'TEXAS_THCIC_ID', 'TEXAS_TOTAL_CHARGES']}
 
    attribute_idx, attribute_dict, max_attr_vals = np.load(datapath_sensfeat,allow_pickle=True)
    col_flags['TEXAS_TOTAL_CHARGES'] = attribute_idx['TOTAL_CHARGES']
    col_flags['TEXAS_RACE'] = attribute_idx['RACE']
    col_flags['TEXAS_ETHN'] = attribute_idx['ETHNICITY']
    col_flags['TEXAS_THCIC_ID'] = None if 'THCIC_ID' not in attribute_idx else attribute_idx['THCIC_ID']
    return np.array([attribute_idx['SEX_CODE'], attribute_idx['RACE'], attribute_idx['ETHNICITY'], attribute_idx['TYPE_OF_ADMISSION']][:size]), attribute_dict, max_attr_vals, col_flags
 


def test(model, loader, args):
    model.eval()
    correct = 0
    with torch.no_grad():
        for info in loader:
            if len(info) == 2:
                data, target = info[0].to(args.device), info[1].to(args.device)
            else:
                data, target, sattr = info[0].to(args.device), info[1].to(args.device), info[2].to(args.device)
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
        for batch_idx, info in enumerate(trainloader):
            if len(info) == 2:
                data, target = info[0].to(args.device), info[1].to(args.device)
            else:
                data, target, sattr = info[0].to(args.device), info[1].to(args.device), info[2].to(args.device)
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



class TexasClassifier(nn.Module):
    def __init__(self,num_classes=100):
        super(TexasClassifier, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(5268,1024),
            nn.Tanh(),
            nn.Linear(1024,1024),
            nn.Tanh(),
            nn.Linear(1024,512),
            nn.Tanh(),
            nn.Linear(512,256),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(256,num_classes)

    def forward(self,x):
        hidden_out = self.features(x)
        return self.classifier(hidden_out)

def main(args: argparse.Namespace) -> None:

    raw_path: Path = Path(args.raw_path)

    num_classes = 100
    train_dataset, test_dataset = load_data(raw_path, args)

    model = TexasClassifier(num_classes=num_classes).to(args.device)

    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    if args.load == "False":

        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)

        train(args.epochs, model, trainloader, testloader, optimizer, args)
        if args.save:
            model_path = "texas.pkl"
            torch.save(model.state_dict(), model_path)
            print("Model Saved!!")

    else:
        model = TexasClassifier(num_classes=num_classes).to(args.device)
        model_path = "texas.pkl"
        model.load_state_dict(torch.load(model_path))
        model.eval()
        # evaluation of model on members
        test(model, trainloader, args)
        # evaluation of model on non-members
        test(model, testloader, args)


def handle_args() -> argparse.Namespace:

    parser: argparse.ArgumentParser = argparse.ArgumentParser()

    parser.add_argument("--raw_path", type = str, default = "./", help = "Location to store raw datasets.")
    parser.add_argument("--lr", type = float, default = 1e-4, help = "Learning Rate")
    parser.add_argument("--decay", type = float, default = 0, help = "Weight decay/L2 Regularization")
    parser.add_argument("--batch_size", type = int, default = 128, help = "Batch size for training data")
    parser.add_argument("--device", type = str, default = torch.device('cuda' if torch.cuda.is_available() else 'cpu'), help = "GPU/CPU")
    parser.add_argument("--save", type = bool, default = True, help = "Save model")
    parser.add_argument("--epochs", type = int, default = 200, help = "Number of Model Training Iterations")
    parser.add_argument("--load", type = str, default = "False", help = "Load trained model")
    args: argparse.Namespace = parser.parse_args()
    return args


if __name__ == "__main__":

    args = handle_args()
    if (Path(args.raw_path) / "Texas100X").exists():
        msg: str = "Datasets exist."
        print(msg)
    else:
        msg: str = "Configuring datasets."
        print(msg)
        args.raw_path = configure(args.raw_path)
        
    main(args)

