import os
import time
import json
import random
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import transformers
from typing import Tuple
from IPython import display
from torch.utils.data import Dataset, DataLoader
from scipy import stats as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import label_binarize

from src.models import RNNInceptionTime, InceptionTimeRNN
from tsai.models.InceptionTime import InceptionTime
from tsai.models.RNN import RNN
from tsai.models.MLP import MLP

class TorchDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        assert len(self.X) == len(self.y), "Lengths of datasets must match"
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class TorchDatasetForForecastingAnomaly(Dataset):
    def __init__(self, X, y, labels):
        self.X = X
        self.y = y
        self.labels = labels
        assert len(self.X) == len(self.y) == len(self.labels), "Lengths of datasets must match"
    
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.labels[idx]


def prepare_dataloaders(X_train: np.ndarray,
                        y_train: np.ndarray,
                        X_test: np.ndarray,
                        y_test: np.ndarray,
                        batch_size: int) -> Tuple[DataLoader, DataLoader]:
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = TorchDataset(X_train_tensor, y_train_tensor)
    test_dataset = TorchDataset(X_test_tensor, y_test_tensor)
    
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)
    
    return train_dataloader, test_dataloader

def split_anomaly_detection_data_idxs(X, y, train_split=0.8): # I still include the X for the parameter just for the consistency
    y_binary = np.where(y > 1, 1, 0) # Convert multi-classes to binary classes (0: normal, 1: anomaly)

    idxs_normal = np.where(y_binary == 0)[0] # Get the indexes of normal data for training and testing
    idxs_anomaly = np.where(y_binary == 1)[0] # Get the indexes of anomaly data for testing only

    print("num normal person data:", len(idxs_normal))
    print("num anomaly person data:", len(idxs_anomaly))
    print("total data:", len(idxs_normal) + len(idxs_anomaly))
    print()

    # Shuffle the indexes of normal and anomaly data
    np.random.shuffle(idxs_normal)
    np.random.shuffle(idxs_anomaly)

    # Split the indexes of normal data for training and testing
    # Use the length of the anomaly data as the length of the normal testing data
    # This is because the anomaly data will be combined with the normal testing data to create a mixed testing data
    # Since the anomaly data is small, the size of the testing sets should not exceed the size of the anomaly data to balance the mixed testing data
    idxs_normal_train_len = int(len(idxs_normal) * train_split)
    idxs_normal_test_len = len(idxs_normal) - idxs_normal_train_len
    idxs_anomaly_test_len = idxs_normal_test_len

    np.random.shuffle(idxs_normal)
    idxs_normal_train = idxs_normal[:idxs_normal_train_len]
    idxs_normal_test = idxs_normal[-idxs_normal_test_len:]

    idxs_anomaly_test = np.random.choice(idxs_anomaly, size=idxs_anomaly_test_len, replace=False)

    # Combine the indexes of normal testing data with anomaly testing data to create a mixed testing data
    idxs_mixed_test = np.concatenate((idxs_normal_test, idxs_anomaly_test))

    # Rename the variables just for convenience
    idxs_train = idxs_normal_train
    idxs_test = idxs_mixed_test

    print("num train data (in person not windows):", len(idxs_train))
    print("num test data (in person not windows):", len(idxs_test))
    print("total data:", len(idxs_train) + len(idxs_test))
    print()
    
    return idxs_train, idxs_test

def prepare_X_dec_and_y(y: torch.Tensor, label_len: int, pred_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    y = y.detach().cpu()
    
    # Split the y into label and pred
    label = y[:, :label_len, :]
    pred = y[:, -pred_len:, :]
    
    # Create decoder input (X_dec)
    # X_dec = label + pred_empty
    pred_empty = torch.zeros([y.shape[0], pred_len, y.shape[-1]], dtype=torch.float32)
    X_dec = torch.cat((label, pred_empty), dim=1)
    
    # Use the pred as the y
    y = pred
    
    return X_dec, y

# def prepare_dataloaders_for_forecasting_anomaly(X_train: np.ndarray, 
#                                                 y_train: np.ndarray, 
#                                                 labels_train: np.ndarray, 
#                                                 X_test: np.ndarray, 
#                                                 y_test: np.ndarray, 
#                                                 labels_test: np.ndarray, 
#                                                 batch_size: int) -> Tuple[DataLoader, DataLoader]:
#     X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
#     y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
#     labels_train_tensor = torch.tensor(labels_train, dtype=torch.float32)
#     X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
#     y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
#     labels_test_tensor = torch.tensor(labels_test, dtype=torch.float32)

#     train_dataset = TorchDatasetForForecastingAnomaly(X_train_tensor, y_train_tensor, labels_train_tensor)
#     test_dataset = TorchDatasetForForecastingAnomaly(X_test_tensor, y_test_tensor, labels_test_tensor)
    
#     train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True)
#     test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)
    
#     return train_dataloader, test_dataloader

def calculate_eval_metrics(pred_one_hot, gt_one_hot, n_data):
    tp = torch.sum((pred_one_hot == 1) & (gt_one_hot == 1), dim=0)
    fp = torch.sum((pred_one_hot == 1) & (gt_one_hot == 0), dim=0)
    fn = torch.sum((pred_one_hot == 0) & (gt_one_hot == 1), dim=0)
    tn = torch.sum((pred_one_hot == 0) & (gt_one_hot == 0), dim=0)

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    specificity = tn / (tn + fp + 1e-10)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-10)
    
    avg_accuracy = accuracy.sum().item() / n_data
    avg_precision = precision / n_data
    avg_recall = recall / n_data
    avg_specificity = specificity / n_data
    avg_f1_score = f1_score / n_data
    
    return avg_accuracy, avg_precision, avg_recall, avg_specificity, avg_f1_score

def print_h(h, nl=128):
    line = "=" * nl
    h_center = h.center(len(line))
    
    print(line)
    print(h_center)
    print(line)

def check_dataset_info(X, y):
    assert X.shape[0] == y.shape[0], f"Mismatch between number of samples in X ({X.shape[0]}) and y ({y.shape[0]})"
    print(f"Total data: {len(X)}")
    print("Label counts:")
    print(pd.DataFrame(y).value_counts().sort_index().reset_index(drop=True))

def calculate_anomaly_score(X):
    q3_list = pd.DataFrame(X).describe().loc['75%'].to_numpy()
    max_list = pd.DataFrame(X).describe().loc['max'].to_numpy()
    return sum(max_list - q3_list)

def calculate_max_mode(X):
    max_list = pd.DataFrame(X).describe().loc['max'].to_numpy()
    return st.mode(max_list).mode, st.mode(max_list).count

def plot_vgrf_data(X_i, title, save_path=None, show=True):
    plt.figure(figsize=(20, 5))
    plt.plot(X_i)
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()

def check_vgrf_data(X, y, note=None, start_idx=None, end_idx=None, limit=10, save_dir=None, show=True):
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for i, (X_i, y_i) in enumerate(zip(X, y)):
        if show:
            display.clear_output(wait=True)
        
        print("================================================================")
        print(f"CHECKING VGRF DATA {i+1} of {min(limit, len(X))}")
        print("================================================================")
        if note:
            print("Note:", note)
            print()
        
        
        print("Full length:", len(X_i))
        if start_idx or end_idx:
            print(f"Sliced length ({start_idx} - {end_idx}):", len(X_i[start_idx:end_idx]))
        print()
        
        # print("Descriptive statistics:")
        # print(pd.DataFrame(X_i[start_idx:end_idx]).describe())
        # print(X_i)
        
        # print("Anomaly score:", calculate_anomaly_score(X_i))
        print("Maximum mode:", calculate_max_mode(X_i))
        
        print('save_path =', os.path.join(save_dir, f'{i}.png'))
        plot_vgrf_data(X_i[start_idx:end_idx], f"VGRF Data with Label {y_i.item()}", save_path=os.path.join(save_dir, f'{i}.png'), show=show)
        
        time.sleep(2)
        
        if i == limit-1:
            break

def get_vgrf_window_data(X, y, window_size: int, stride_size_: int, zeros_filter_thres: int) -> Tuple[torch.Tensor, torch.Tensor]:
    X_window_list = []
    y_list = []
    person_idxs = []
    
    for person_idx, X_person in enumerate(X):
        for idx in range(0, X_person.shape[0] - window_size, stride_size_):
            X_window = X_person[idx:idx+window_size, :]  
            
            # Skip the window if some rows are zeros
            X_window_no_zeros = X_window[~torch.all(X_window == 0, dim=1)]
            no_zeros_thres = zeros_filter_thres
            if X_window_no_zeros.shape[0] < int(window_size * no_zeros_thres):
                # if X_window_no_zeros.shape[0] > 0:
                #     display.clear_output()
                #     plot_vgrf_data(X_window, y[person_idx])
                #     time.sleep(2)
                continue
            
            # zero_pct = (X_window == 0).sum().item() / (X_window.shape[0] * X_window.shape[1])
            # if zero_pct > 0.555:
                # if zero_pct != 1.0 and zero_pct <= 0.6:
                #     display.clear_output()
                #     print(zero_pct)
                #     plot_vgrf_data(X_window, y[person_idx])
                #     time.sleep(2)
                # continue
            
            # Skip the window If mode of the maximums occurs more than 75% in the features
            # W, F = X_window.shape
            # max_mode, max_mode_count = calculate_max_mode(X_window)
            # if max_mode_count > (F * 0.75):
            #     display.clear_output()
            #     plot_vgrf_data(X_window, y[person_idx])
            #     time.sleep(2)
            #     continue
            
            # Skip anomaly data with max_mode == 0.5
            # if max_mode == 0.5:
            #     continue
            
            X_window_list.append(X_window)
            y_list.append(y[person_idx])
            person_idxs.append(person_idx)
        
    vgrf_window_data_tensor = torch.tensor(np.array(X_window_list, dtype=np.float32), dtype=torch.float32)
    
    # if vgrf_window_data_tensor.shape[1] < window_size:
    #     padding_tensor = torch.zeros((window_size - vgrf_window_data_tensor.shape[0], vgrf_window_data_tensor.shape[1]))
    #     vgrf_window_data_tensor = torch.cat((vgrf_window_data_tensor, padding_tensor), dim=0)
    
    y_tensor = torch.tensor(y_list, dtype=torch.long)
    person_idx_tensor = torch.tensor(person_idxs, dtype=torch.long)
    return vgrf_window_data_tensor, y_tensor, person_idx_tensor
        
def eval_window(model, dataloader, criterion=None, average='weighted'):
    device = next(iter(model.parameters())).device
    
    # Switch the model to evaluation mode
    model.eval()

    loss_list = [] if criterion else None
    y_gt_list = []
    y_pred_list = []

    with torch.no_grad():
        for X, y in dataloader:
            X = X.permute(0, 2, 1)
            X = X.to(device)
            y = y.to(device)

            # Feed forward
            y_pred = model(X)
            
            # Calculate the validation loss
            if criterion:
                loss = criterion(y_pred, y)
                loss_list.append(loss.item())
            
            # Calculate metrics
            _, y_pred = torch.max(y_pred, 1)

            y_gt_list += y.tolist()
            y_pred_list += y_pred.tolist()
    
    avg_loss = sum(loss_list)/len(loss_list) if loss_list else None
    acc = accuracy_score(y_gt_list, y_pred_list)
    f1 = f1_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    precision = precision_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    recall = recall_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    cm = confusion_matrix(y_gt_list, y_pred_list).tolist()
    return avg_loss, acc, f1, precision, recall, cm

# def eval_person(model, dataset, window_size, stride_size, zeros_filter_thres=1.0, criterion=None, average='weighted', debug=False):
def eval_person(model, dataset, window_size, zeros_filter_thres=1.0, criterion=None, average='weighted', debug=False):
    stride_size = window_size
    device = next(iter(model.parameters())).device
    
    # Switch the model to evaluation mode
    model.eval()

    loss_list = [] if criterion else None
    y_gt_list = []
    y_pred_list = []

    with torch.no_grad():
        for i, (X_person, y_person) in enumerate(dataset):
            X_window, y_window, _ = get_vgrf_window_data(X_person.unsqueeze(0), 
                                                         y_person.unsqueeze(0).tolist(), 
                                                         window_size, 
                                                         stride_size,
                                                         zeros_filter_thres)
            if len(X_window) == 0:
                continue
            
            X_window = X_window.permute(0, 2, 1)
            X_window = X_window.to(device)
            y_window = y_window.to(device)
            
            # Feed forward
            y_pred = model(X_window)
            
            # Calculate the validation loss
            if criterion:
                loss = criterion(y_pred, y_window)
                loss_list.append(loss.item())
            
            # Calculate metrics
            y_pred_probs, y_pred_labels = torch.max(torch.nn.functional.softmax(y_pred, dim=1), dim=1) # Logits to labels
            # !!!
            
            # If there's a label that is not 0 in the predicted labels
            # then the (final) predicted label is not 0 (Healthy)
            if (y_pred_labels > 0).any():
                y_pred, count = torch.mode(y_pred_labels[y_pred_labels > 0]) # Get the most frequent label as the predicted label
                y_pred = y_pred.item()
            else:    
                y_pred = 0
            
            if debug:
                print(f"person: {(i+1):02d}, "
                      f"y_pred: {[f'{l} ({(p*100):.2f}%)' for l, p in zip(y_pred_labels.tolist(), y_pred_probs.tolist())]} -> {y_pred}, "
                      f"y_gt: {y_person.item()}, " 
                      f"correct: {y_pred == y_person.item()}")
                if i+1 == len(dataset):
                    print()
            
            y_pred_list.append(y_pred)
            y_gt_list.append(y_person.item())
            
    avg_loss = sum(loss_list)/len(loss_list) if loss_list else None
    acc = accuracy_score(y_gt_list, y_pred_list)
    f1 = f1_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    precision = precision_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    recall = recall_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    cm = confusion_matrix(y_gt_list, y_pred_list).tolist()
    return avg_loss, acc, f1, precision, recall, cm

# def eval_person_with_thres(model, dataset, window_size, stride_size, zeros_filter_thres=1.0, criterion=None, average='weighted', thres=0.5, debug=False):
def eval_person_with_thres(model, dataset, window_size, zeros_filter_thres=1.0, criterion=None, average='weighted', thres=0.5, debug=False):
    stride_size = window_size
    device = next(iter(model.parameters())).device
    
    # Switch the model to evaluation mode
    model.eval()

    loss_list = [] if criterion else None
    y_gt_list = []
    y_pred_list = []

    with torch.no_grad():
        for i, (X_person, y_person) in enumerate(dataset):
            X_window, y_window, _ = get_vgrf_window_data(X_person.unsqueeze(0), 
                                                         y_person.unsqueeze(0).tolist(), 
                                                         window_size, 
                                                         stride_size,
                                                         zeros_filter_thres)
            if len(X_window) == 0:
                continue
            
            X_window = X_window.permute(0, 2, 1)
            X_window = X_window.to(device)
            y_window = y_window.to(device)
            
            # Feed forward
            y_pred = model(X_window)
            
            # Calculate the validation loss
            if criterion:
                loss = criterion(y_pred, y_window)
                loss_list.append(loss.item())
            
            # Calculate metrics
            y_pred_probs, y_pred_labels = torch.max(torch.nn.functional.softmax(y_pred, dim=1), dim=1) # Logits to labels
            # !!!

            cond = (y_pred_probs > thres).int()
            y_pred_labels = y_pred_labels * cond
            
            # If there's a label that is not 0 in the predicted labels
            # then the (final) predicted label is not 0 (Healthy)
            if (y_pred_labels > 0).any():
                y_pred, count = torch.mode(y_pred_labels[y_pred_labels > 0]) # Get the most frequent label as the predicted label
                y_pred = y_pred.item()
            else:    
                y_pred = 0
            
            if debug:
                print(f"person: {(i+1):02d}, "
                      f"y_pred: {[f'{l} ({(p*100):.2f}%)' for l, p in zip(y_pred_labels.tolist(), y_pred_probs.tolist())]} -> {y_pred}, "
                      f"y_gt: {y_person.item()}, " 
                      f"correct: {y_pred == y_person.item()}")
                if i+1 == len(dataset):
                    print()
            
            y_pred_list.append(y_pred)
            y_gt_list.append(y_person.item())
            
    avg_loss = sum(loss_list)/len(loss_list) if loss_list else None
    acc = accuracy_score(y_gt_list, y_pred_list)
    f1 = f1_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    precision = precision_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    recall = recall_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    cm = confusion_matrix(y_gt_list, y_pred_list).tolist()
    return avg_loss, acc, f1, precision, recall, cm

# def eval_person_severity_voting(model, dataset, window_size, stride_size, zeros_filter_thres=1.0, criterion=None, average='weighted', debug=False):
def eval_person_severity_voting(model, dataset, window_size, zeros_filter_thres=1.0, criterion=None, average='weighted', debug=False):
    stride_size = window_size
    device = next(iter(model.parameters())).device
    
    # Switch the model to evaluation mode
    model.eval()

    loss_list = [] if criterion else None
    y_gt_list = []
    y_pred_list = []

    with torch.no_grad():
        for i, (X_person, y_person) in enumerate(dataset):
            X_window, y_window, _ = get_vgrf_window_data(X_person.unsqueeze(0), 
                                                         y_person.unsqueeze(0).tolist(), 
                                                         window_size, 
                                                         stride_size,
                                                         zeros_filter_thres)
            if len(X_window) == 0:
                continue
            
            X_window = X_window.permute(0, 2, 1)
            X_window = X_window.to(device)
            y_window = y_window.to(device)
            
            # Feed forward
            y_pred = model(X_window)
            
            # Calculate the validation loss
            if criterion:
                loss = criterion(y_pred, y_window)
                loss_list.append(loss.item())
            
            # Calculate metrics
            y_pred_probs, y_pred_labels = torch.max(torch.nn.functional.softmax(y_pred, dim=1), dim=1) # Logits to labels
            # !!!
            
            # If there's a label that is not 0 in the predicted labels
            # then the (final) predicted label is not 0 (Healthy)
            if (y_pred_labels > 0).any():
                # y_pred, count = torch.mode(y_pred_labels[y_pred_labels > 0]) # Get the most frequent label as the predicted label
                y_pred = torch.bincount(y_pred_labels[y_pred_labels > 0]).argmax()
                y_pred = y_pred.item()
            else:    
                y_pred = 0
            
            if debug:
                print(f"person: {(i+1):02d}, "
                      f"y_pred: {[f'{l} ({(p*100):.2f}%)' for l, p in zip(y_pred_labels.tolist(), y_pred_probs.tolist())]} -> {y_pred}, "
                      f"y_gt: {y_person.item()}, " 
                      f"correct: {y_pred == y_person.item()}")
                if i+1 == len(dataset):
                    print()
            
            y_pred_list.append(y_pred)
            y_gt_list.append(y_person.item())
            
    avg_loss = sum(loss_list)/len(loss_list) if loss_list else None
    acc = accuracy_score(y_gt_list, y_pred_list)
    f1 = f1_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    precision = precision_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    recall = recall_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    cm = confusion_matrix(y_gt_list, y_pred_list).tolist()
    return avg_loss, acc, f1, precision, recall, cm

# def eval_person_majority_voting(model, dataset, window_size, stride_size, zeros_filter_thres=1.0, criterion=None, average='weighted', debug=False):
def eval_person_majority_voting(model, dataset, window_size, zeros_filter_thres=1.0, criterion=None, average='weighted', debug=False):
    stride_size = window_size
    device = next(iter(model.parameters())).device
    
    # Switch the model to evaluation mode
    model.eval()

    loss_list = [] if criterion else None
    y_gt_list = []
    y_pred_list = []
    y_pred_score_list = []
    y_pred_scores_list = []

    with torch.no_grad():
        for i, (X_person, y_person) in enumerate(dataset):
            X_window, y_window, _ = get_vgrf_window_data(X_person.unsqueeze(0), 
                                                         y_person.unsqueeze(0).tolist(), 
                                                         window_size, 
                                                         stride_size,
                                                         zeros_filter_thres)
            if len(X_window) == 0:
                continue
            
            X_window = X_window.permute(0, 2, 1)
            X_window = X_window.to(device)
            y_window = y_window.to(device)
            
            # Feed forward
            y_pred = model(X_window)
            
            # Calculate the validation loss
            if criterion:
                loss = criterion(y_pred, y_window)
                loss_list.append(loss.item())
            
            # Calculate metrics
            y_pred_probs_all = torch.nn.functional.softmax(y_pred, dim=1) # Logits to labels
            y_pred_probs_avg = y_pred_probs_all.mean(dim=0)
            y_pred_probs, y_pred_labels = torch.max(y_pred_probs_all, dim=1)

            # y_pred, count = torch.mode(y_pred_labels) # Get the most frequent label as the predicted label
            y_pred = torch.bincount(y_pred_labels).argmax()
            y_pred = y_pred.item()

            # Calculate confident score
            y_pred_probs, y_pred_labels = y_pred_probs.cpu().numpy(), y_pred_labels.cpu().numpy()

            y_pred_labels_correct = np.where(y_pred_labels == y_pred, 1, 0)
            y_pred_probs_correct = y_pred_probs * y_pred_labels_correct # TODO: Check if there is zero in y_pred_probs_correct
            y_pred_score = np.mean(y_pred_probs_correct).item()
            if y_pred == 0:
                y_pred_score = 1.0 - y_pred_score
            
            if debug:
                print(f"person: {(i+1):02d}, "
                      f"y_pred: {[f'{l} ({(p*100):.2f}%)' for l, p in zip(y_pred_labels.tolist(), y_pred_probs.tolist())]} -> {y_pred}, "
                      f"y_gt: {y_person.item()}, " 
                      f"correct: {y_pred == y_person.item()}")
                if i+1 == len(dataset):
                    print()
            
            y_pred_list.append(y_pred)
            y_gt_list.append(y_person.item())
            y_pred_score_list.append(y_pred_score)
            y_pred_scores_list.append(y_pred_probs_avg.tolist())
            
    avg_loss = sum(loss_list)/len(loss_list) if loss_list else None
    acc = accuracy_score(y_gt_list, y_pred_list)
    f1 = f1_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    precision = precision_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    recall = recall_score(y_gt_list, y_pred_list, average=average, zero_division=0)

    # Calculate ROC AUC (multi-class)
    y_pred_scores_list = np.array(y_pred_scores_list)
    n_label = y_pred_scores_list.shape[1]
    y_gt_list_binary = label_binarize(y_gt_list, classes=list(range(n_label)))

    fpr_multiclass_list = []
    tpr_multiclass_list = []
    roc_auc_multiclass_list = []

    for i in range(n_label):
        # Skip class if no positive instances
        if np.sum(y_gt_list_binary[:, i]) == 0:
            continue

        fpr_multiclass, tpr_multiclass, _ = roc_curve(y_gt_list_binary[:, i], y_pred_scores_list[:, i])
        roc_auc_multiclass = auc(fpr_multiclass, tpr_multiclass)

        fpr_multiclass_list.append(fpr_multiclass.tolist())
        tpr_multiclass_list.append(tpr_multiclass.tolist())
        roc_auc_multiclass_list.append(roc_auc_multiclass.item())
    
    roc_auc_multiclass_avg = np.mean(roc_auc_multiclass_list).item()

    # Calculate ROC AUC (binary)
    y_gt_list_binary = np.where(np.array(y_gt_list) == 0, 0, 1)
    fpr_binary, tpr_binary, _ = roc_curve(y_gt_list_binary, y_pred_score_list)
    fpr_binary, tpr_binary = fpr_binary.tolist(), tpr_binary.tolist()
    roc_auc_binary = auc(fpr_binary, tpr_binary).item()

    # Calculate confusion matrix
    cm = confusion_matrix(y_gt_list, y_pred_list, labels=list(range(n_label))).tolist()

    return (
        avg_loss, acc, f1, precision, recall, cm, 
        # ROC AUC metrics (binary)
        fpr_binary, tpr_binary, roc_auc_binary,
        # ROC AUC metrics (multi-class)
        fpr_multiclass_list, tpr_multiclass_list, roc_auc_multiclass_list, roc_auc_multiclass_avg
    )

# def eval_person_max_severity(model, dataset, window_size, stride_size, zeros_filter_thres=1.0, criterion=None, average='weighted', debug=False):
def eval_person_max_severity(model, dataset, window_size, zeros_filter_thres=1.0, criterion=None, average='weighted', debug=False):
    stride_size = window_size
    device = next(iter(model.parameters())).device
    
    # Switch the model to evaluation mode
    model.eval()

    loss_list = [] if criterion else None
    y_gt_list = []
    y_pred_list = []

    with torch.no_grad():
        for i, (X_person, y_person) in enumerate(dataset):
            X_window, y_window, _ = get_vgrf_window_data(X_person.unsqueeze(0), 
                                                         y_person.unsqueeze(0).tolist(), 
                                                         window_size, 
                                                         stride_size,
                                                         zeros_filter_thres)
            if len(X_window) == 0:
                continue
            
            X_window = X_window.permute(0, 2, 1)
            X_window = X_window.to(device)
            y_window = y_window.to(device)
            
            # Feed forward
            y_pred = model(X_window)
            
            # Calculate the validation loss
            if criterion:
                loss = criterion(y_pred, y_window)
                loss_list.append(loss.item())
            
            # Calculate metrics
            y_pred_probs, y_pred_labels = torch.max(torch.nn.functional.softmax(y_pred, dim=1), dim=1) # Logits to labels
            # !!!
            
            y_pred = y_pred_labels.max().item()
            
            if debug:
                print(f"person: {(i+1):02d}, "
                      f"y_pred: {[f'{l} ({(p*100):.2f}%)' for l, p in zip(y_pred_labels.tolist(), y_pred_probs.tolist())]} -> {y_pred}, "
                      f"y_gt: {y_person.item()}, " 
                      f"correct: {y_pred == y_person.item()}")
                if i+1 == len(dataset):
                    print()
            
            y_pred_list.append(y_pred)
            y_gt_list.append(y_person.item())
            
    avg_loss = sum(loss_list)/len(loss_list) if loss_list else None
    acc = accuracy_score(y_gt_list, y_pred_list)
    f1 = f1_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    precision = precision_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    recall = recall_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    cm = confusion_matrix(y_gt_list, y_pred_list).tolist()
    return avg_loss, acc, f1, precision, recall, cm

# Pos
def eval_window_pos(model, dataloader, criterion=None, average='weighted'):
    device = next(iter(model.parameters())).device
    
    # Switch the model to evaluation mode
    model.eval()

    loss_list = [] if criterion else None
    y_gt_list = []
    y_pred_list = []

    with torch.no_grad():
        for X, X_pos, y in dataloader:
            X = X.permute(0, 2, 1)
            X_pos = X_pos.permute(0, 2, 1)
            X = X.to(device)
            X_pos = X_pos.to(device)
            y = y.to(device)

            # Feed forward
            y_pred = model(X, X_pos)
            
            # Calculate the validation loss
            if criterion:
                loss = criterion(y_pred, y)
                loss_list.append(loss.item())
            
            # Calculate metrics
            _, y_pred = torch.max(y_pred, 1)

            y_gt_list += y.tolist()
            y_pred_list += y_pred.tolist()
    
    avg_loss = sum(loss_list)/len(loss_list) if loss_list else None
    acc = accuracy_score(y_gt_list, y_pred_list)
    f1 = f1_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    precision = precision_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    recall = recall_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    cm = confusion_matrix(y_gt_list, y_pred_list).tolist()
    return avg_loss, acc, f1, precision, recall, cm

def eval_person_pos(model, dataset, window_size, stride_size, zeros_filter_thres=1.0, criterion=None, average='weighted', debug=False):
    device = next(iter(model.parameters())).device
    
    # Switch the model to evaluation mode
    model.eval()

    loss_list = [] if criterion else None
    y_gt_list = []
    y_pred_list = []

    with torch.no_grad():
        for i, (X_person, X_pos_person, y_person) in enumerate(dataset):
            X_window, y_window, _ = get_vgrf_window_data(X_person.unsqueeze(0), 
                                                         y_person.unsqueeze(0).tolist(), 
                                                         window_size, 
                                                         stride_size,
                                                         zeros_filter_thres)
            X_pos_window, y_window, _ = get_vgrf_window_data(X_pos_person.unsqueeze(0), 
                                                             y_person.unsqueeze(0).tolist(), 
                                                             window_size, 
                                                             stride_size,
                                                             zeros_filter_thres)
            if len(X_window) == 0:
                continue
            if len(X_pos_window) == 0:
                continue
            
            X_window = X_window.permute(0, 2, 1)
            X_pos_window = X_pos_window.permute(0, 2, 1)
            X_window = X_window.to(device)
            X_pos_window = X_pos_window.to(device)
            y_window = y_window.to(device)
            
            # Feed forward
            y_pred = model(X_window, X_pos_window)
            
            # Calculate the validation loss
            if criterion:
                loss = criterion(y_pred, y_window)
                loss_list.append(loss.item())
            
            # Calculate metrics
            y_pred_probs, y_pred_labels = torch.max(torch.nn.functional.softmax(y_pred, dim=1), dim=1) # Logits to labels
            # !!!
            
            # If there's a label that is not 0 in the predicted labels
            # then the (final) predicted label is not 0 (Healthy)
            if (y_pred_labels > 0).any():
                y_pred, count = torch.mode(y_pred_labels[y_pred_labels > 0]) # Get the most frequent label as the predicted label
                y_pred = y_pred.item()
            else:    
                y_pred = 0
            
            if debug:
                print(f"person: {(i+1):02d}, "
                      f"y_pred: {[f'{l} ({(p*100):.2f}%)' for l, p in zip(y_pred_labels.tolist(), y_pred_probs.tolist())]} -> {y_pred}, "
                      f"y_gt: {y_person.item()}, " 
                      f"correct: {y_pred == y_person.item()}")
                if i+1 == len(dataset):
                    print()
            
            y_pred_list.append(y_pred)
            y_gt_list.append(y_person.item())
            
    avg_loss = sum(loss_list)/len(loss_list) if loss_list else None
    acc = accuracy_score(y_gt_list, y_pred_list)
    f1 = f1_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    precision = precision_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    recall = recall_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    cm = confusion_matrix(y_gt_list, y_pred_list).tolist()
    return avg_loss, acc, f1, precision, recall, cm

def eval_person_with_thres_pos(model, dataset, window_size, stride_size, zeros_filter_thres=1.0, criterion=None, average='weighted', thres=0.5, debug=False):
    device = next(iter(model.parameters())).device
    
    # Switch the model to evaluation mode
    model.eval()

    loss_list = [] if criterion else None
    y_gt_list = []
    y_pred_list = []

    with torch.no_grad():
        for i, (X_person, X_pos_person, y_person) in enumerate(dataset):
            X_window, y_window, _ = get_vgrf_window_data(X_person.unsqueeze(0), 
                                                         y_person.unsqueeze(0).tolist(), 
                                                         window_size, 
                                                         stride_size,
                                                         zeros_filter_thres)
            X_pos_window, y_window, _ = get_vgrf_window_data(X_pos_person.unsqueeze(0), 
                                                             y_person.unsqueeze(0).tolist(), 
                                                             window_size, 
                                                             stride_size,
                                                             zeros_filter_thres)
            if len(X_window) == 0:
                continue
            if len(X_pos_window) == 0:
                continue
            
            X_window = X_window.permute(0, 2, 1)
            X_pos_window = X_pos_window.permute(0, 2, 1)
            X_window = X_window.to(device)
            X_pos_window = X_pos_window.to(device)
            y_window = y_window.to(device)
            
            # Feed forward
            y_pred = model(X_window, X_pos_window)
            
            # Calculate the validation loss
            if criterion:
                loss = criterion(y_pred, y_window)
                loss_list.append(loss.item())
            
            # Calculate metrics
            y_pred_probs, y_pred_labels = torch.max(torch.nn.functional.softmax(y_pred, dim=1), dim=1) # Logits to labels
            # !!!

            cond = (y_pred_probs > thres).int()
            y_pred_labels = y_pred_labels * cond
            
            # If there's a label that is not 0 in the predicted labels
            # then the (final) predicted label is not 0 (Healthy)
            if (y_pred_labels > 0).any():
                y_pred, count = torch.mode(y_pred_labels[y_pred_labels > 0]) # Get the most frequent label as the predicted label
                y_pred = y_pred.item()
            else:    
                y_pred = 0
            
            if debug:
                print(f"person: {(i+1):02d}, "
                      f"y_pred: {[f'{l} ({(p*100):.2f}%)' for l, p in zip(y_pred_labels.tolist(), y_pred_probs.tolist())]} -> {y_pred}, "
                      f"y_gt: {y_person.item()}, " 
                      f"correct: {y_pred == y_person.item()}")
                if i+1 == len(dataset):
                    print()
            
            y_pred_list.append(y_pred)
            y_gt_list.append(y_person.item())
            
    avg_loss = sum(loss_list)/len(loss_list) if loss_list else None
    acc = accuracy_score(y_gt_list, y_pred_list)
    f1 = f1_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    precision = precision_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    recall = recall_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    cm = confusion_matrix(y_gt_list, y_pred_list).tolist()
    return avg_loss, acc, f1, precision, recall, cm

def eval_person_severity_voting_pos(model, dataset, window_size, stride_size, zeros_filter_thres=1.0, criterion=None, average='weighted', debug=False):
    device = next(iter(model.parameters())).device
    
    # Switch the model to evaluation mode
    model.eval()

    loss_list = [] if criterion else None
    y_gt_list = []
    y_pred_list = []

    with torch.no_grad():
        for i, (X_person, X_pos_person, y_person) in enumerate(dataset):
            X_window, y_window, _ = get_vgrf_window_data(X_person.unsqueeze(0), 
                                                         y_person.unsqueeze(0).tolist(), 
                                                         window_size, 
                                                         stride_size,
                                                         zeros_filter_thres)
            X_pos_window, y_window, _ = get_vgrf_window_data(X_pos_person.unsqueeze(0), 
                                                             y_person.unsqueeze(0).tolist(), 
                                                             window_size, 
                                                             stride_size,
                                                             zeros_filter_thres)
            if len(X_window) == 0:
                continue
            if len(X_pos_window) == 0:
                continue
            
            X_window = X_window.permute(0, 2, 1)
            X_pos_window = X_pos_window.permute(0, 2, 1)
            X_window = X_window.to(device)
            X_pos_window = X_pos_window.to(device)
            y_window = y_window.to(device)
            
            # Feed forward
            y_pred = model(X_window, X_pos_window)
            
            # Calculate the validation loss
            if criterion:
                loss = criterion(y_pred, y_window)
                loss_list.append(loss.item())
            
            # Calculate metrics
            y_pred_probs, y_pred_labels = torch.max(torch.nn.functional.softmax(y_pred, dim=1), dim=1) # Logits to labels
            # !!!
            
            # If there's a label that is not 0 in the predicted labels
            # then the (final) predicted label is not 0 (Healthy)
            if (y_pred_labels > 0).any():
                y_pred, count = torch.mode(y_pred_labels[y_pred_labels > 0]) # Get the most frequent label as the predicted label
                y_pred = y_pred.item()
            else:    
                y_pred = 0
            
            if debug:
                print(f"person: {(i+1):02d}, "
                      f"y_pred: {[f'{l} ({(p*100):.2f}%)' for l, p in zip(y_pred_labels.tolist(), y_pred_probs.tolist())]} -> {y_pred}, "
                      f"y_gt: {y_person.item()}, " 
                      f"correct: {y_pred == y_person.item()}")
                if i+1 == len(dataset):
                    print()
            
            y_pred_list.append(y_pred)
            y_gt_list.append(y_person.item())
            
    avg_loss = sum(loss_list)/len(loss_list) if loss_list else None
    acc = accuracy_score(y_gt_list, y_pred_list)
    f1 = f1_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    precision = precision_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    recall = recall_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    cm = confusion_matrix(y_gt_list, y_pred_list).tolist()
    return avg_loss, acc, f1, precision, recall, cm

def eval_person_majority_voting_pos(model, dataset, window_size, stride_size, zeros_filter_thres=1.0, criterion=None, average='weighted', debug=False):
    device = next(iter(model.parameters())).device
    
    # Switch the model to evaluation mode
    model.eval()

    loss_list = [] if criterion else None
    y_gt_list = []
    y_pred_list = []

    with torch.no_grad():
        for i, (X_person, X_pos_person, y_person) in enumerate(dataset):
            X_window, y_window, _ = get_vgrf_window_data(X_person.unsqueeze(0), 
                                                         y_person.unsqueeze(0).tolist(), 
                                                         window_size, 
                                                         stride_size,
                                                         zeros_filter_thres)
            X_pos_window, y_window, _ = get_vgrf_window_data(X_pos_person.unsqueeze(0), 
                                                             y_person.unsqueeze(0).tolist(), 
                                                             window_size, 
                                                             stride_size,
                                                             zeros_filter_thres)
            if len(X_window) == 0:
                continue
            if len(X_pos_window) == 0:
                continue
            
            X_window = X_window.permute(0, 2, 1)
            X_pos_window = X_pos_window.permute(0, 2, 1)
            X_window = X_window.to(device)
            X_pos_window = X_pos_window.to(device)
            y_window = y_window.to(device)
            
            # Feed forward
            y_pred = model(X_window, X_pos_window)
            
            # Calculate the validation loss
            if criterion:
                loss = criterion(y_pred, y_window)
                loss_list.append(loss.item())
            
            # Calculate metrics
            y_pred_probs, y_pred_labels = torch.max(torch.nn.functional.softmax(y_pred, dim=1), dim=1) # Logits to labels
            # !!!
            
            y_pred, count = torch.mode(y_pred_labels) # Get the most frequent label as the predicted label
            y_pred = y_pred.item()
            
            if debug:
                print(f"person: {(i+1):02d}, "
                      f"y_pred: {[f'{l} ({(p*100):.2f}%)' for l, p in zip(y_pred_labels.tolist(), y_pred_probs.tolist())]} -> {y_pred}, "
                      f"y_gt: {y_person.item()}, " 
                      f"correct: {y_pred == y_person.item()}")
                if i+1 == len(dataset):
                    print()
            
            y_pred_list.append(y_pred)
            y_gt_list.append(y_person.item())
            
    avg_loss = sum(loss_list)/len(loss_list) if loss_list else None
    acc = accuracy_score(y_gt_list, y_pred_list)
    f1 = f1_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    precision = precision_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    recall = recall_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    cm = confusion_matrix(y_gt_list, y_pred_list).tolist()
    return avg_loss, acc, f1, precision, recall, cm

def eval_person_max_severity_pos(model, dataset, window_size, stride_size, zeros_filter_thres=1.0, criterion=None, average='weighted', debug=False):
    device = next(iter(model.parameters())).device
    
    # Switch the model to evaluation mode
    model.eval()

    loss_list = [] if criterion else None
    y_gt_list = []
    y_pred_list = []

    with torch.no_grad():
        for i, (X_person, X_pos_person, y_person) in enumerate(dataset):
            X_window, y_window, _ = get_vgrf_window_data(X_person.unsqueeze(0), 
                                                         y_person.unsqueeze(0).tolist(), 
                                                         window_size, 
                                                         stride_size,
                                                         zeros_filter_thres)
            X_pos_window, y_window, _ = get_vgrf_window_data(X_pos_person.unsqueeze(0), 
                                                             y_person.unsqueeze(0).tolist(), 
                                                             window_size, 
                                                             stride_size,
                                                             zeros_filter_thres)
            if len(X_window) == 0:
                continue
            if len(X_pos_window) == 0:
                continue
            
            X_window = X_window.permute(0, 2, 1)
            X_pos_window = X_pos_window.permute(0, 2, 1)
            X_window = X_window.to(device)
            X_pos_window = X_pos_window.to(device)
            y_window = y_window.to(device)
            
            # Feed forward
            y_pred = model(X_window, X_pos_window)
            
            # Calculate the validation loss
            if criterion:
                loss = criterion(y_pred, y_window)
                loss_list.append(loss.item())
            
            # Calculate metrics
            y_pred_probs, y_pred_labels = torch.max(torch.nn.functional.softmax(y_pred, dim=1), dim=1) # Logits to labels
            # !!!
            
            y_pred = y_pred_labels.max().item()
            
            if debug:
                print(f"person: {(i+1):02d}, "
                      f"y_pred: {[f'{l} ({(p*100):.2f}%)' for l, p in zip(y_pred_labels.tolist(), y_pred_probs.tolist())]} -> {y_pred}, "
                      f"y_gt: {y_person.item()}, " 
                      f"correct: {y_pred == y_person.item()}")
                if i+1 == len(dataset):
                    print()
            
            y_pred_list.append(y_pred)
            y_gt_list.append(y_person.item())
            
    avg_loss = sum(loss_list)/len(loss_list) if loss_list else None
    acc = accuracy_score(y_gt_list, y_pred_list)
    f1 = f1_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    precision = precision_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    recall = recall_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    cm = confusion_matrix(y_gt_list, y_pred_list).tolist()
    return avg_loss, acc, f1, precision, recall, cm

# Body
def get_vgrf_window_data_body(X, X_body, y, window_size: int, stride_size_: int, zeros_filter_thres: int) -> Tuple[torch.Tensor, torch.Tensor]:
    X_window_list = []
    X_body_list = []
    y_list = []
    person_idxs = []
    
    for person_idx, X_person in enumerate(X):
        for idx in range(0, X_person.shape[0] - window_size, stride_size_):
            X_window = X_person[idx:idx+window_size, :]  
            
            # Skip the window if some rows are zeros
            X_window_no_zeros = X_window[~torch.all(X_window == 0, dim=1)]
            no_zeros_thres = zeros_filter_thres
            if X_window_no_zeros.shape[0] < int(window_size * no_zeros_thres):
                # if X_window_no_zeros.shape[0] > 0:
                #     display.clear_output()
                #     plot_vgrf_data(X_window, y[person_idx])
                #     time.sleep(2)
                continue
            
            # zero_pct = (X_window == 0).sum().item() / (X_window.shape[0] * X_window.shape[1])
            # if zero_pct > 0.555:
                # if zero_pct != 1.0 and zero_pct <= 0.6:
                #     display.clear_output()
                #     print(zero_pct)
                #     plot_vgrf_data(X_window, y[person_idx])
                #     time.sleep(2)
                # continue
            
            # Skip the window If mode of the maximums occurs more than 75% in the features
            # W, F = X_window.shape
            # max_mode, max_mode_count = calculate_max_mode(X_window)
            # if max_mode_count > (F * 0.75):
            #     display.clear_output()
            #     plot_vgrf_data(X_window, y[person_idx])
            #     time.sleep(2)
            #     continue
            
            # Skip anomaly data with max_mode == 0.5
            # if max_mode == 0.5:
            #     continue
            
            X_window_list.append(X_window)
            X_body_list.append(X_body[person_idx])
            y_list.append(y[person_idx])
            person_idxs.append(person_idx)
        
    vgrf_window_data_tensor = torch.tensor(np.array(X_window_list, dtype=np.float32), dtype=torch.float32)
    
    # if vgrf_window_data_tensor.shape[1] < window_size:
    #     padding_tensor = torch.zeros((window_size - vgrf_window_data_tensor.shape[0], vgrf_window_data_tensor.shape[1]))
    #     vgrf_window_data_tensor = torch.cat((vgrf_window_data_tensor, padding_tensor), dim=0)
    print(f'{X_body_list}')
    X_body_tensor = torch.tensor(X_body_list)
    y_tensor = torch.tensor(y_list, dtype=torch.long)
    person_idx_tensor = torch.tensor(person_idxs, dtype=torch.long)
    return vgrf_window_data_tensor, X_body_tensor, y_tensor, person_idx_tensor

def eval_window_body(model, dataloader, criterion=None, average='weighted'):
    device = next(iter(model.parameters())).device
    
    # Switch the model to evaluation mode
    model.eval()

    loss_list = [] if criterion else None
    y_gt_list = []
    y_pred_list = []

    with torch.no_grad():
        for X, X_body, y in dataloader:
            X = X.permute(0, 2, 1)
            X = X.to(device)
            X_body = X_body.to(device)
            y = y.to(device)

            # Feed forward
            y_pred = model(X, X_body)
            
            # Calculate the validation loss
            if criterion:
                loss = criterion(y_pred, y)
                loss_list.append(loss.item())
            
            # Calculate metrics
            _, y_pred = torch.max(y_pred, 1)

            y_gt_list += y.tolist()
            y_pred_list += y_pred.tolist()
    
    avg_loss = sum(loss_list)/len(loss_list) if loss_list else None
    acc = accuracy_score(y_gt_list, y_pred_list)
    f1 = f1_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    precision = precision_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    recall = recall_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    cm = confusion_matrix(y_gt_list, y_pred_list).tolist()
    return avg_loss, acc, f1, precision, recall, cm

def eval_person_body(model, dataset, window_size, stride_size, zeros_filter_thres=1.0, criterion=None, average='weighted', debug=False):
    device = next(iter(model.parameters())).device
    
    # Switch the model to evaluation mode
    model.eval()

    loss_list = [] if criterion else None
    y_gt_list = []
    y_pred_list = []

    with torch.no_grad():
        for i, (X_person, X_body_person, y_person) in enumerate(dataset):
            X_window, X_body_window, y_window, _ = get_vgrf_window_data_body(X_person.unsqueeze(0), 
                                                                             X_body_person.unsqueeze(0).tolist(),
                                                                             y_person.unsqueeze(0).tolist(), 
                                                                             window_size, 
                                                                             stride_size,
                                                                             zeros_filter_thres)
            if len(X_window) == 0:
                continue
            if len(X_body_window) == 0:
                continue
            
            X_window = X_window.permute(0, 2, 1)
            X_window = X_window.to(device)
            X_body_window = X_body_window.to(device)
            y_window = y_window.to(device)
            
            # Feed forward
            y_pred = model(X_window, X_body_window)
            
            # Calculate the validation loss
            if criterion:
                loss = criterion(y_pred, y_window)
                loss_list.append(loss.item())
            
            # Calculate metrics
            y_pred_probs, y_pred_labels = torch.max(torch.nn.functional.softmax(y_pred, dim=1), dim=1) # Logits to labels
            # !!!
            
            # If there's a label that is not 0 in the predicted labels
            # then the (final) predicted label is not 0 (Healthy)
            if (y_pred_labels > 0).any():
                y_pred, count = torch.mode(y_pred_labels[y_pred_labels > 0]) # Get the most frequent label as the predicted label
                y_pred = y_pred.item()
            else:    
                y_pred = 0
            
            if debug:
                print(f"person: {(i+1):02d}, "
                      f"y_pred: {[f'{l} ({(p*100):.2f}%)' for l, p in zip(y_pred_labels.tolist(), y_pred_probs.tolist())]} -> {y_pred}, "
                      f"y_gt: {y_person.item()}, " 
                      f"correct: {y_pred == y_person.item()}")
                if i+1 == len(dataset):
                    print()
            
            y_pred_list.append(y_pred)
            y_gt_list.append(y_person.item())
            
    avg_loss = sum(loss_list)/len(loss_list) if loss_list else None
    acc = accuracy_score(y_gt_list, y_pred_list)
    f1 = f1_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    precision = precision_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    recall = recall_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    cm = confusion_matrix(y_gt_list, y_pred_list).tolist()
    return avg_loss, acc, f1, precision, recall, cm

def eval_person_with_thres_body(model, dataset, window_size, stride_size, zeros_filter_thres=1.0, criterion=None, average='weighted', thres=0.5, debug=False):
    device = next(iter(model.parameters())).device
    
    # Switch the model to evaluation mode
    model.eval()

    loss_list = [] if criterion else None
    y_gt_list = []
    y_pred_list = []

    with torch.no_grad():
        for i, (X_person, X_body_person, y_person) in enumerate(dataset):
            X_window, X_body_window, y_window, _ = get_vgrf_window_data_body(X_person.unsqueeze(0), 
                                                                             X_body_person.unsqueeze(0).tolist(),
                                                                             y_person.unsqueeze(0).tolist(), 
                                                                             window_size, 
                                                                             stride_size,
                                                                             zeros_filter_thres)
            if len(X_window) == 0:
                continue
            if len(X_body_window) == 0:
                continue
            
            X_window = X_window.permute(0, 2, 1)
            X_window = X_window.to(device)
            X_body_window = X_body_window.to(device)
            y_window = y_window.to(device)
            
            # Feed forward
            y_pred = model(X_window, X_body_window)
            
            # Calculate the validation loss
            if criterion:
                loss = criterion(y_pred, y_window)
                loss_list.append(loss.item())
            
            # Calculate metrics
            y_pred_probs, y_pred_labels = torch.max(torch.nn.functional.softmax(y_pred, dim=1), dim=1) # Logits to labels
            # !!!

            cond = (y_pred_probs > thres).int()
            y_pred_labels = y_pred_labels * cond
            
            # If there's a label that is not 0 in the predicted labels
            # then the (final) predicted label is not 0 (Healthy)
            if (y_pred_labels > 0).any():
                y_pred, count = torch.mode(y_pred_labels[y_pred_labels > 0]) # Get the most frequent label as the predicted label
                y_pred = y_pred.item()
            else:    
                y_pred = 0
            
            if debug:
                print(f"person: {(i+1):02d}, "
                      f"y_pred: {[f'{l} ({(p*100):.2f}%)' for l, p in zip(y_pred_labels.tolist(), y_pred_probs.tolist())]} -> {y_pred}, "
                      f"y_gt: {y_person.item()}, " 
                      f"correct: {y_pred == y_person.item()}")
                if i+1 == len(dataset):
                    print()
            
            y_pred_list.append(y_pred)
            y_gt_list.append(y_person.item())
            
    avg_loss = sum(loss_list)/len(loss_list) if loss_list else None
    acc = accuracy_score(y_gt_list, y_pred_list)
    f1 = f1_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    precision = precision_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    recall = recall_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    cm = confusion_matrix(y_gt_list, y_pred_list).tolist()
    return avg_loss, acc, f1, precision, recall, cm

def eval_person_severity_voting_body(model, dataset, window_size, stride_size, zeros_filter_thres=1.0, criterion=None, average='weighted', debug=False):
    device = next(iter(model.parameters())).device
    
    # Switch the model to evaluation mode
    model.eval()

    loss_list = [] if criterion else None
    y_gt_list = []
    y_pred_list = []

    with torch.no_grad():
        for i, (X_person, X_body_person, y_person) in enumerate(dataset):
            X_window, X_body_window, y_window, _ = get_vgrf_window_data_body(X_person.unsqueeze(0), 
                                                                             X_body_person.unsqueeze(0).tolist(),
                                                                             y_person.unsqueeze(0).tolist(), 
                                                                             window_size, 
                                                                             stride_size,
                                                                             zeros_filter_thres)
            if len(X_window) == 0:
                continue
            if len(X_body_window) == 0:
                continue
            
            X_window = X_window.permute(0, 2, 1)
            X_window = X_window.to(device)
            X_body_window = X_body_window.to(device)
            y_window = y_window.to(device)
            
            # Feed forward
            print(f'{X_body_window.shape=}')
            y_pred = model(X_window, X_body_window)
            
            # Calculate the validation loss
            if criterion:
                loss = criterion(y_pred, y_window)
                loss_list.append(loss.item())
            
            # Calculate metrics
            y_pred_probs, y_pred_labels = torch.max(torch.nn.functional.softmax(y_pred, dim=1), dim=1) # Logits to labels
            # !!!
            
            # If there's a label that is not 0 in the predicted labels
            # then the (final) predicted label is not 0 (Healthy)
            if (y_pred_labels > 0).any():
                y_pred, count = torch.mode(y_pred_labels[y_pred_labels > 0]) # Get the most frequent label as the predicted label
                y_pred = y_pred.item()
            else:    
                y_pred = 0
            
            if debug:
                print(f"person: {(i+1):02d}, "
                      f"y_pred: {[f'{l} ({(p*100):.2f}%)' for l, p in zip(y_pred_labels.tolist(), y_pred_probs.tolist())]} -> {y_pred}, "
                      f"y_gt: {y_person.item()}, " 
                      f"correct: {y_pred == y_person.item()}")
                if i+1 == len(dataset):
                    print()
            
            y_pred_list.append(y_pred)
            y_gt_list.append(y_person.item())
            
    avg_loss = sum(loss_list)/len(loss_list) if loss_list else None
    acc = accuracy_score(y_gt_list, y_pred_list)
    f1 = f1_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    precision = precision_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    recall = recall_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    cm = confusion_matrix(y_gt_list, y_pred_list).tolist()
    return avg_loss, acc, f1, precision, recall, cm

def eval_person_majority_voting_body(model, dataset, window_size, stride_size, zeros_filter_thres=1.0, criterion=None, average='weighted', debug=False):
    device = next(iter(model.parameters())).device
    
    # Switch the model to evaluation mode
    model.eval()

    loss_list = [] if criterion else None
    y_gt_list = []
    y_pred_list = []

    with torch.no_grad():
        for i, (X_person, X_body_person, y_person) in enumerate(dataset):
            X_window, X_body_window, y_window, _ = get_vgrf_window_data_body(X_person.unsqueeze(0), 
                                                                             X_body_person.unsqueeze(0).tolist(),
                                                                             y_person.unsqueeze(0).tolist(), 
                                                                             window_size, 
                                                                             stride_size,
                                                                             zeros_filter_thres)
            if len(X_window) == 0:
                continue
            if len(X_body_window) == 0:
                continue
            
            X_window = X_window.permute(0, 2, 1)
            X_window = X_window.to(device)
            X_body_window = X_body_window.to(device)
            y_window = y_window.to(device)
            
            # Feed forward
            y_pred = model(X_window, X_body_window)
            
            # Calculate the validation loss
            if criterion:
                loss = criterion(y_pred, y_window)
                loss_list.append(loss.item())
            
            # Calculate metrics
            y_pred_probs, y_pred_labels = torch.max(torch.nn.functional.softmax(y_pred, dim=1), dim=1) # Logits to labels
            # !!!
            
            y_pred, count = torch.mode(y_pred_labels) # Get the most frequent label as the predicted label
            y_pred = y_pred.item()
            
            if debug:
                print(f"person: {(i+1):02d}, "
                      f"y_pred: {[f'{l} ({(p*100):.2f}%)' for l, p in zip(y_pred_labels.tolist(), y_pred_probs.tolist())]} -> {y_pred}, "
                      f"y_gt: {y_person.item()}, " 
                      f"correct: {y_pred == y_person.item()}")
                if i+1 == len(dataset):
                    print()
            
            y_pred_list.append(y_pred)
            y_gt_list.append(y_person.item())
            
    avg_loss = sum(loss_list)/len(loss_list) if loss_list else None
    acc = accuracy_score(y_gt_list, y_pred_list)
    f1 = f1_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    precision = precision_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    recall = recall_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    cm = confusion_matrix(y_gt_list, y_pred_list).tolist()
    return avg_loss, acc, f1, precision, recall, cm

def eval_person_max_severity_body(model, dataset, window_size, stride_size, zeros_filter_thres=1.0, criterion=None, average='weighted', debug=False):
    device = next(iter(model.parameters())).device
    
    # Switch the model to evaluation mode
    model.eval()

    loss_list = [] if criterion else None
    y_gt_list = []
    y_pred_list = []

    with torch.no_grad():
        for i, (X_person, X_body_person, y_person) in enumerate(dataset):
            X_window, X_body_window, y_window, _ = get_vgrf_window_data_body(X_person.unsqueeze(0), 
                                                                             X_body_person.unsqueeze(0).tolist(),
                                                                             y_person.unsqueeze(0).tolist(), 
                                                                             window_size, 
                                                                             stride_size,
                                                                             zeros_filter_thres)
            if len(X_window) == 0:
                continue
            if len(X_body_window) == 0:
                continue
            
            X_window = X_window.permute(0, 2, 1)
            X_window = X_window.to(device)
            X_body_window = X_body_window.to(device)
            y_window = y_window.to(device)
            
            # Feed forward
            y_pred = model(X_window, X_body_window)
            
            # Calculate the validation loss
            if criterion:
                loss = criterion(y_pred, y_window)
                loss_list.append(loss.item())
            
            # Calculate metrics
            y_pred_probs, y_pred_labels = torch.max(torch.nn.functional.softmax(y_pred, dim=1), dim=1) # Logits to labels
            # !!!
            
            y_pred = y_pred_labels.max().item()
            
            if debug:
                print(f"person: {(i+1):02d}, "
                      f"y_pred: {[f'{l} ({(p*100):.2f}%)' for l, p in zip(y_pred_labels.tolist(), y_pred_probs.tolist())]} -> {y_pred}, "
                      f"y_gt: {y_person.item()}, " 
                      f"correct: {y_pred == y_person.item()}")
                if i+1 == len(dataset):
                    print()
            
            y_pred_list.append(y_pred)
            y_gt_list.append(y_person.item())
            
    avg_loss = sum(loss_list)/len(loss_list) if loss_list else None
    acc = accuracy_score(y_gt_list, y_pred_list)
    f1 = f1_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    precision = precision_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    recall = recall_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    cm = confusion_matrix(y_gt_list, y_pred_list).tolist()
    return avg_loss, acc, f1, precision, recall, cm

# XFX
def get_vgrf_window_data_xfx(X, X_xfx, y, window_size: int, stride_size_: int, zeros_filter_thres: int) -> Tuple[torch.Tensor, torch.Tensor]:
    X_window_list = []
    X_xfx_list = []
    y_list = []
    person_idxs = []
    
    for person_idx, X_person in enumerate(X):
        for idx in range(0, X_person.shape[0] - window_size, stride_size_):
            X_window = X_person[idx:idx+window_size, :]  
            
            # Skip the window if some rows are zeros
            X_window_no_zeros = X_window[~torch.all(X_window == 0, dim=1)]
            no_zeros_thres = zeros_filter_thres
            if X_window_no_zeros.shape[0] < int(window_size * no_zeros_thres):
                # if X_window_no_zeros.shape[0] > 0:
                #     display.clear_output()
                #     plot_vgrf_data(X_window, y[person_idx])
                #     time.sleep(2)
                continue
            
            # zero_pct = (X_window == 0).sum().item() / (X_window.shape[0] * X_window.shape[1])
            # if zero_pct > 0.555:
                # if zero_pct != 1.0 and zero_pct <= 0.6:
                #     display.clear_output()
                #     print(zero_pct)
                #     plot_vgrf_data(X_window, y[person_idx])
                #     time.sleep(2)
                # continue
            
            # Skip the window If mode of the maximums occurs more than 75% in the features
            # W, F = X_window.shape
            # max_mode, max_mode_count = calculate_max_mode(X_window)
            # if max_mode_count > (F * 0.75):
            #     display.clear_output()
            #     plot_vgrf_data(X_window, y[person_idx])
            #     time.sleep(2)
            #     continue
            
            # Skip anomaly data with max_mode == 0.5
            # if max_mode == 0.5:
            #     continue
            
            X_window_list.append(X_window)
            X_xfx_list.append(X_xfx[person_idx])
            y_list.append(y[person_idx])
            person_idxs.append(person_idx)
        
    vgrf_window_data_tensor = torch.tensor(np.array(X_window_list, dtype=np.float32), dtype=torch.float32)
    
    # if vgrf_window_data_tensor.shape[1] < window_size:
    #     padding_tensor = torch.zeros((window_size - vgrf_window_data_tensor.shape[0], vgrf_window_data_tensor.shape[1]))
    #     vgrf_window_data_tensor = torch.cat((vgrf_window_data_tensor, padding_tensor), dim=0)
    
    X_xfx_tensor = torch.tensor(X_xfx_list)
    y_tensor = torch.tensor(y_list, dtype=torch.long)
    person_idx_tensor = torch.tensor(person_idxs, dtype=torch.long)
    return vgrf_window_data_tensor, X_xfx_tensor, y_tensor, person_idx_tensor

def eval_window_xfx(model, dataloader, criterion=None, average='weighted'):
    device = next(iter(model.parameters())).device
    
    # Switch the model to evaluation mode
    model.eval()

    loss_list = [] if criterion else None
    y_gt_list = []
    y_pred_list = []

    with torch.no_grad():
        for X, X_xfx, y in dataloader:
            X = X.permute(0, 2, 1)
            X = X.to(device)
            X_xfx = X_xfx.to(device)
            y = y.to(device)

            # Feed forward
            y_pred = model(X, X_xfx)
            
            # Calculate the validation loss
            if criterion:
                loss = criterion(y_pred, y)
                loss_list.append(loss.item())
            
            # Calculate metrics
            _, y_pred = torch.max(y_pred, 1)

            y_gt_list += y.tolist()
            y_pred_list += y_pred.tolist()
    
    avg_loss = sum(loss_list)/len(loss_list) if loss_list else None
    acc = accuracy_score(y_gt_list, y_pred_list)
    f1 = f1_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    precision = precision_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    recall = recall_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    cm = confusion_matrix(y_gt_list, y_pred_list).tolist()
    return avg_loss, acc, f1, precision, recall, cm

def eval_person_xfx(model, dataset, window_size, stride_size, zeros_filter_thres=1.0, criterion=None, average='weighted', debug=False):
    device = next(iter(model.parameters())).device
    
    # Switch the model to evaluation mode
    model.eval()

    loss_list = [] if criterion else None
    y_gt_list = []
    y_pred_list = []

    with torch.no_grad():
        for i, (X_person, X_xfx_person, y_person) in enumerate(dataset):
            X_window, X_xfx_window, y_window, _ = get_vgrf_window_data_xfx(X_person.unsqueeze(0), 
                                                                             X_xfx_person.unsqueeze(0).tolist(),
                                                                             y_person.unsqueeze(0).tolist(), 
                                                                             window_size, 
                                                                             stride_size,
                                                                             zeros_filter_thres)
            if len(X_window) == 0:
                continue
            if len(X_xfx_window) == 0:
                continue
            
            X_window = X_window.permute(0, 2, 1)
            X_window = X_window.to(device)
            X_xfx_window = X_xfx_window.to(device)
            y_window = y_window.to(device)
            
            # Feed forward
            y_pred = model(X_window, X_xfx_window)
            
            # Calculate the validation loss
            if criterion:
                loss = criterion(y_pred, y_window)
                loss_list.append(loss.item())
            
            # Calculate metrics
            y_pred_probs, y_pred_labels = torch.max(torch.nn.functional.softmax(y_pred, dim=1), dim=1) # Logits to labels
            # !!!
            
            # If there's a label that is not 0 in the predicted labels
            # then the (final) predicted label is not 0 (Healthy)
            if (y_pred_labels > 0).any():
                y_pred, count = torch.mode(y_pred_labels[y_pred_labels > 0]) # Get the most frequent label as the predicted label
                y_pred = y_pred.item()
            else:    
                y_pred = 0
            
            if debug:
                print(f"person: {(i+1):02d}, "
                      f"y_pred: {[f'{l} ({(p*100):.2f}%)' for l, p in zip(y_pred_labels.tolist(), y_pred_probs.tolist())]} -> {y_pred}, "
                      f"y_gt: {y_person.item()}, " 
                      f"correct: {y_pred == y_person.item()}")
                if i+1 == len(dataset):
                    print()
            
            y_pred_list.append(y_pred)
            y_gt_list.append(y_person.item())
            
    avg_loss = sum(loss_list)/len(loss_list) if loss_list else None
    acc = accuracy_score(y_gt_list, y_pred_list)
    f1 = f1_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    precision = precision_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    recall = recall_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    cm = confusion_matrix(y_gt_list, y_pred_list).tolist()
    return avg_loss, acc, f1, precision, recall, cm

def eval_person_with_thres_xfx(model, dataset, window_size, stride_size, zeros_filter_thres=1.0, criterion=None, average='weighted', thres=0.5, debug=False):
    device = next(iter(model.parameters())).device
    
    # Switch the model to evaluation mode
    model.eval()

    loss_list = [] if criterion else None
    y_gt_list = []
    y_pred_list = []

    with torch.no_grad():
        for i, (X_person, X_xfx_person, y_person) in enumerate(dataset):
            X_window, X_xfx_window, y_window, _ = get_vgrf_window_data_xfx(X_person.unsqueeze(0), 
                                                                             X_xfx_person.unsqueeze(0).tolist(),
                                                                             y_person.unsqueeze(0).tolist(), 
                                                                             window_size, 
                                                                             stride_size,
                                                                             zeros_filter_thres)
            if len(X_window) == 0:
                continue
            if len(X_xfx_window) == 0:
                continue
            
            X_window = X_window.permute(0, 2, 1)
            X_window = X_window.to(device)
            X_xfx_window = X_xfx_window.to(device)
            y_window = y_window.to(device)
            
            # Feed forward
            y_pred = model(X_window, X_xfx_window)
            
            # Calculate the validation loss
            if criterion:
                loss = criterion(y_pred, y_window)
                loss_list.append(loss.item())
            
            # Calculate metrics
            y_pred_probs, y_pred_labels = torch.max(torch.nn.functional.softmax(y_pred, dim=1), dim=1) # Logits to labels
            # !!!

            cond = (y_pred_probs > thres).int()
            y_pred_labels = y_pred_labels * cond
            
            # If there's a label that is not 0 in the predicted labels
            # then the (final) predicted label is not 0 (Healthy)
            if (y_pred_labels > 0).any():
                y_pred, count = torch.mode(y_pred_labels[y_pred_labels > 0]) # Get the most frequent label as the predicted label
                y_pred = y_pred.item()
            else:    
                y_pred = 0
            
            if debug:
                print(f"person: {(i+1):02d}, "
                      f"y_pred: {[f'{l} ({(p*100):.2f}%)' for l, p in zip(y_pred_labels.tolist(), y_pred_probs.tolist())]} -> {y_pred}, "
                      f"y_gt: {y_person.item()}, " 
                      f"correct: {y_pred == y_person.item()}")
                if i+1 == len(dataset):
                    print()
            
            y_pred_list.append(y_pred)
            y_gt_list.append(y_person.item())
            
    avg_loss = sum(loss_list)/len(loss_list) if loss_list else None
    acc = accuracy_score(y_gt_list, y_pred_list)
    f1 = f1_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    precision = precision_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    recall = recall_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    cm = confusion_matrix(y_gt_list, y_pred_list).tolist()
    return avg_loss, acc, f1, precision, recall, cm

def eval_person_severity_voting_xfx(model, dataset, window_size, stride_size, zeros_filter_thres=1.0, criterion=None, average='weighted', debug=False):
    device = next(iter(model.parameters())).device
    
    # Switch the model to evaluation mode
    model.eval()

    loss_list = [] if criterion else None
    y_gt_list = []
    y_pred_list = []

    with torch.no_grad():
        for i, (X_person, X_xfx_person, y_person) in enumerate(dataset):
            X_window, X_xfx_window, y_window, _ = get_vgrf_window_data_xfx(X_person.unsqueeze(0), 
                                                                             X_xfx_person.unsqueeze(0).tolist(),
                                                                             y_person.unsqueeze(0).tolist(), 
                                                                             window_size, 
                                                                             stride_size,
                                                                             zeros_filter_thres)
            if len(X_window) == 0:
                continue
            if len(X_xfx_window) == 0:
                continue
            
            X_window = X_window.permute(0, 2, 1)
            X_window = X_window.to(device)
            X_xfx_window = X_xfx_window.to(device)
            y_window = y_window.to(device)
            
            # Feed forward
            y_pred = model(X_window, X_xfx_window)
            
            # Calculate the validation loss
            if criterion:
                loss = criterion(y_pred, y_window)
                loss_list.append(loss.item())
            
            # Calculate metrics
            y_pred_probs, y_pred_labels = torch.max(torch.nn.functional.softmax(y_pred, dim=1), dim=1) # Logits to labels
            # !!!
            
            # If there's a label that is not 0 in the predicted labels
            # then the (final) predicted label is not 0 (Healthy)
            if (y_pred_labels > 0).any():
                y_pred, count = torch.mode(y_pred_labels[y_pred_labels > 0]) # Get the most frequent label as the predicted label
                y_pred = y_pred.item()
            else:    
                y_pred = 0
            
            if debug:
                print(f"person: {(i+1):02d}, "
                      f"y_pred: {[f'{l} ({(p*100):.2f}%)' for l, p in zip(y_pred_labels.tolist(), y_pred_probs.tolist())]} -> {y_pred}, "
                      f"y_gt: {y_person.item()}, " 
                      f"correct: {y_pred == y_person.item()}")
                if i+1 == len(dataset):
                    print()
            
            y_pred_list.append(y_pred)
            y_gt_list.append(y_person.item())
            
    avg_loss = sum(loss_list)/len(loss_list) if loss_list else None
    acc = accuracy_score(y_gt_list, y_pred_list)
    f1 = f1_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    precision = precision_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    recall = recall_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    cm = confusion_matrix(y_gt_list, y_pred_list).tolist()
    return avg_loss, acc, f1, precision, recall, cm

def eval_person_majority_voting_xfx(model, dataset, window_size, stride_size, zeros_filter_thres=1.0, criterion=None, average='weighted', debug=False):
    device = next(iter(model.parameters())).device
    
    # Switch the model to evaluation mode
    model.eval()

    loss_list = [] if criterion else None
    y_gt_list = []
    y_pred_list = []

    with torch.no_grad():
        for i, (X_person, X_xfx_person, y_person) in enumerate(dataset):
            X_window, X_xfx_window, y_window, _ = get_vgrf_window_data_xfx(X_person.unsqueeze(0), 
                                                                             X_xfx_person.unsqueeze(0).tolist(),
                                                                             y_person.unsqueeze(0).tolist(), 
                                                                             window_size, 
                                                                             stride_size,
                                                                             zeros_filter_thres)
            if len(X_window) == 0:
                continue
            if len(X_xfx_window) == 0:
                continue
            
            X_window = X_window.permute(0, 2, 1)
            X_window = X_window.to(device)
            X_xfx_window = X_xfx_window.to(device)
            y_window = y_window.to(device)
            
            # Feed forward
            y_pred = model(X_window, X_xfx_window)
            
            # Calculate the validation loss
            if criterion:
                loss = criterion(y_pred, y_window)
                loss_list.append(loss.item())
            
            # Calculate metrics
            y_pred_probs, y_pred_labels = torch.max(torch.nn.functional.softmax(y_pred, dim=1), dim=1) # Logits to labels
            # !!!
            
            y_pred, count = torch.mode(y_pred_labels) # Get the most frequent label as the predicted label
            y_pred = y_pred.item()
            
            if debug:
                print(f"person: {(i+1):02d}, "
                      f"y_pred: {[f'{l} ({(p*100):.2f}%)' for l, p in zip(y_pred_labels.tolist(), y_pred_probs.tolist())]} -> {y_pred}, "
                      f"y_gt: {y_person.item()}, " 
                      f"correct: {y_pred == y_person.item()}")
                if i+1 == len(dataset):
                    print()
            
            y_pred_list.append(y_pred)
            y_gt_list.append(y_person.item())
            
    avg_loss = sum(loss_list)/len(loss_list) if loss_list else None
    acc = accuracy_score(y_gt_list, y_pred_list)
    f1 = f1_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    precision = precision_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    recall = recall_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    cm = confusion_matrix(y_gt_list, y_pred_list).tolist()
    return avg_loss, acc, f1, precision, recall, cm

def eval_person_max_severity_xfx(model, dataset, window_size, stride_size, zeros_filter_thres=1.0, criterion=None, average='weighted', debug=False):
    device = next(iter(model.parameters())).device
    
    # Switch the model to evaluation mode
    model.eval()

    loss_list = [] if criterion else None
    y_gt_list = []
    y_pred_list = []

    with torch.no_grad():
        for i, (X_person, X_xfx_person, y_person) in enumerate(dataset):
            X_window, X_xfx_window, y_window, _ = get_vgrf_window_data_xfx(X_person.unsqueeze(0), 
                                                                             X_xfx_person.unsqueeze(0).tolist(),
                                                                             y_person.unsqueeze(0).tolist(), 
                                                                             window_size, 
                                                                             stride_size,
                                                                             zeros_filter_thres)
            if len(X_window) == 0:
                continue
            if len(X_xfx_window) == 0:
                continue
            
            X_window = X_window.permute(0, 2, 1)
            X_window = X_window.to(device)
            X_xfx_window = X_xfx_window.to(device)
            y_window = y_window.to(device)
            
            # Feed forward
            y_pred = model(X_window, X_xfx_window)
            
            # Calculate the validation loss
            if criterion:
                loss = criterion(y_pred, y_window)
                loss_list.append(loss.item())
            
            # Calculate metrics
            y_pred_probs, y_pred_labels = torch.max(torch.nn.functional.softmax(y_pred, dim=1), dim=1) # Logits to labels
            # !!!
            
            y_pred = y_pred_labels.max().item()
            
            if debug:
                print(f"person: {(i+1):02d}, "
                      f"y_pred: {[f'{l} ({(p*100):.2f}%)' for l, p in zip(y_pred_labels.tolist(), y_pred_probs.tolist())]} -> {y_pred}, "
                      f"y_gt: {y_person.item()}, " 
                      f"correct: {y_pred == y_person.item()}")
                if i+1 == len(dataset):
                    print()
            
            y_pred_list.append(y_pred)
            y_gt_list.append(y_person.item())
            
    avg_loss = sum(loss_list)/len(loss_list) if loss_list else None
    acc = accuracy_score(y_gt_list, y_pred_list)
    f1 = f1_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    precision = precision_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    recall = recall_score(y_gt_list, y_pred_list, average=average, zero_division=0)
    cm = confusion_matrix(y_gt_list, y_pred_list).tolist()
    return avg_loss, acc, f1, precision, recall, cm

# ================================================================
# NEW UPDATES
# ================================================================
def set_seed(seed):
    # Set random seed for NumPy
    np.random.seed(seed)

    # Set random seed for Torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True  # Ensures deterministic results
    torch.backends.cudnn.benchmark = False  # Avoids non-deterministic algorithms

    # Set random seed for Transformers
    transformers.set_seed(seed)

    # Optionally set random seed for sklearn and Python's own random module
    random.seed(seed)

    # Set random seed for os
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"Random seed: {seed}")

def init_metrics(metric_names=['acc', 'f1', 'precision', 'recall', 'cm']):
    metrics = {}
    for metric_name in metric_names:
        if metric_name in ['acc', 'f1', 'precision', 'recall', 'roc_auc_multiclass_avg']:
            metrics[metric_name] = {'folds': [], 'avg': None, 'std': None}
        else:
            metrics[metric_name] = {'folds': []}
    return metrics

def update_metrics(metrics, in_metrics):
    for metric_name in in_metrics.keys():
        metrics[metric_name]['folds'] += [in_metrics[metric_name]]
        if metric_name in ['acc', 'f1', 'precision', 'recall', 'roc_auc_multiclass_avg']:
            metrics[metric_name]['avg'] = np.mean(metrics[metric_name]['folds']).item()
            metrics[metric_name]['std'] = np.std(metrics[metric_name]['folds']).item()
    return metrics

def plot_k_fold_metrics_roc_curves(metrics, k_fold, n_col=5):
    # Determine the grid size for subplots
    n_row = (k_fold + n_col - 1) // n_col # Calculate number of rows

    # Create subplots
    fig, axes = plt.subplots(n_row, n_col, figsize=(n_col * 5, n_row * 5))
    axes = axes.flatten() # Flatten the 2D array of axes for easy indexing

    for i_fold in range(k_fold):
        tpr = metrics['tpr']['folds'][i_fold]
        fpr = metrics['fpr']['folds'][i_fold]
        roc_auc = metrics['roc_auc']['folds'][i_fold]

        ax = axes[i_fold]
        ax.plot(fpr, tpr, color='blue', lw=2, label=f"AUC = {roc_auc:.2f}")
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.set_title(f"FOLD {i_fold+1} ROC Curve")
        ax.legend(loc='lower right')
        ax.grid()

    # Remove empty subplots if k_fold < total grid spaces
    for i in range(k_fold, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()

def save_k_fold_metrics_roc_curves(metrics, k_fold):
    for i_fold in range(k_fold):
        # Extract ROC metrics for the current fold
        tpr = metrics['tpr']['folds'][i_fold]
        fpr = metrics['fpr']['folds'][i_fold]
        roc_auc = metrics['roc_auc']['folds'][i_fold]

        # Create a new figure for each fold
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color='blue', lw=2, label=f"AUC = {roc_auc:.2f}")
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"FOLD {i_fold+1} ROC Curve")
        plt.legend(loc='lower right')
        plt.grid()

        # Save the figure separately
        plt.savefig(f'roc_curve_fold_{i_fold+1}.png')
        plt.close() # Close the figure to free memory

    print(f"All {k_fold}-Folds ROC curve figures saved successfully.")

def plot_k_fold_roc_curves_multiclass(fpr_folds, tpr_folds, auc_folds, figsize=(4, 3), save_dir='evaluations/roc_curves_multiclass'):
    k_fold = len(fpr_folds)
    n_class = len(fpr_folds[0])

    # Create subplots: Rows = Folds, Columns = Classes
    fig, axes = plt.subplots(k_fold, n_class, figsize=(figsize[0] * n_class, figsize[1] * k_fold))

    for fold_idx in range(k_fold):
        for class_idx in range(n_class):
            ax = axes[fold_idx, class_idx] if k_fold > 1 else axes[class_idx]
            
            fpr = fpr_folds[fold_idx][class_idx]
            tpr = tpr_folds[fold_idx][class_idx]
            auc = auc_folds[fold_idx][class_idx]
            
            ax.plot(fpr, tpr, color='blue', linestyle='-', lw=2, label=f"AUC = {auc:.3f}")
            ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
            ax.set_title(f"Fold {fold_idx+1} - Class {class_idx} ROC Curve")
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.legend()
            ax.grid()

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, '_fold_all.png')
        plt.savefig(save_path)

    plt.show()

    if save_dir:
        print("Saved in:", save_dir)

def plot_k_fold_roc_curves_multiclass_v2(fpr_folds, tpr_folds, auc_folds, class_names, figsize=(4, 4), save_dir='evaluations/roc_curves_multiclass/v2', show=True, i_folds=None):
    if not i_folds:
        i_folds = [i_fold+1 for i_fold in range(len(fpr_folds))]
    assert len(fpr_folds) == len(tpr_folds) == len(auc_folds) == len(i_folds) # TODO: Add assert message

    k_fold = len(fpr_folds)
    n_class = len(fpr_folds[0])

    # Grid layout: 5 rows x 2 columns (adjust if k_fold < 10)
    n_cols = 5
    n_rows = math.ceil(k_fold / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0] * n_cols, figsize[1] * n_rows))
    axes = axes.flatten()  # Flatten for easier indexing

    for fold_idx in range(k_fold):
        ax = axes[fold_idx]
        for class_idx in range(n_class):
            fpr = fpr_folds[fold_idx][class_idx]
            tpr = tpr_folds[fold_idx][class_idx]
            auc = auc_folds[fold_idx][class_idx]
            ax.plot(fpr, tpr, lw=2, label=f"{class_names[class_idx]} (AUC = {auc:.3f})")

        ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
        ax.set_title(f"ROC Curves - Fold {i_folds[fold_idx]}")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend(loc='lower right')
        ax.grid()

    # Hide any unused subplots
    for i in range(k_fold, len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, '_fold_all.png')
        plt.savefig(save_path)

    if show:
        plt.show()

    if save_dir:
        print("Saved in:", save_dir)

def save_k_fold_roc_curves_multiclass(fpr_folds, tpr_folds, auc_folds, figsize=(4, 3), save_dir='evaluations/roc_curves_multiclass'):
    os.makedirs(save_dir, exist_ok=True)

    k_fold = len(fpr_folds)
    n_class = len(fpr_folds[0])

    for fold_idx in range(k_fold):
        for class_idx in range(n_class):
            fig, ax = plt.subplots(figsize=figsize)
            
            fpr = fpr_folds[fold_idx][class_idx]
            tpr = tpr_folds[fold_idx][class_idx]
            auc = auc_folds[fold_idx][class_idx]
            
            ax.plot(fpr, tpr, color='blue', linestyle='-', lw=2, label=f"AUC = {auc:.3f}")
            ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
            ax.set_title(f"Fold {fold_idx+1} - Class {class_idx} ROC Curve")
            ax.set_xlabel("FPR")
            ax.set_ylabel("TPR")
            ax.legend()
            ax.grid()

            plt.tight_layout()
            save_path = os.path.join(save_dir, f'fold_{fold_idx+1}_class_{class_idx}.png')
            plt.savefig(save_path) # Save the figure separately
            plt.close(fig) # Close the figure to free memory

    print("Saved ROC curves (multi-class) in:", save_dir)

def save_k_fold_roc_curves_multiclass_v2(fpr_folds, tpr_folds, auc_folds, figsize=(4, 4), save_dir='evaluations/roc_curves_multiclass/v2'):
    os.makedirs(save_dir, exist_ok=True)

    k_fold = len(fpr_folds)
    n_class = len(fpr_folds[0])

    for fold_idx in range(k_fold):
        fig, ax = plt.subplots(figsize=figsize)

        for class_idx in range(n_class):
            fpr = fpr_folds[fold_idx][class_idx]
            tpr = tpr_folds[fold_idx][class_idx]
            auc = auc_folds[fold_idx][class_idx]
            ax.plot(fpr, tpr, lw=2, label=f"Class {class_idx} (AUC = {auc:.3f})")

        ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
        ax.set_title(f"ROC Curves - Fold {fold_idx + 1}")
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        ax.legend(loc="lower right")
        ax.grid()

        plt.tight_layout()
        save_path = os.path.join(save_dir, f'fold_{fold_idx+1}.png')
        plt.savefig(save_path) # Save the figure separately
        plt.close(fig) # Close the figure to free memory

    print("Saved ROC curves (multi-class) in:", save_dir)

def plot_k_fold_cm(cm_folds, class_names=None, annot_types=['pct', 'frac'], cbar=True, figsize=(24, 9), save_dir='evaluations/cm', show=True, n_col=5, i_folds=None):
    if not i_folds:
        i_folds = [i_fold+1 for i_fold in range(len(cm_folds))]
    assert len(cm_folds) == len(i_folds) # TODO: Add assert message

    # Convert to NumPy array
    cm_folds = np.array(cm_folds, dtype=float)  # (folds, rows, cols)

    # Normalize by row
    row_sums = cm_folds.sum(axis=2, keepdims=True)
    cm_folds_norm = np.divide(cm_folds, row_sums, where=row_sums != 0)

    # Format annotations
    annot_types = str(annot_types)
    annots = np.empty_like(cm_folds_norm, dtype=object)
    for i_fold in range(cm_folds.shape[0]):
        for j_row in range(cm_folds.shape[1]):
            for k_col in range(cm_folds.shape[2]):
                count = int(cm_folds[i_fold, j_row, k_col])
                total_count = int(row_sums[i_fold, j_row, 0])  # corrected index
                pct = cm_folds_norm[i_fold, j_row, k_col] * 100
                annot = []
                if 'pct' in annot_types:
                    annot.append(f"{pct:.1f}%")
                if 'frac' in annot_types:
                    annot.append(f"{count}/{total_count}")
                annots[i_fold, j_row, k_col] = "\n".join(annot) if annot else str(count)

    # Plot setup
    k_fold = cm_folds.shape[0]
    cols = n_col if n_col else k_fold
    rows = int(np.ceil(k_fold / cols))
    fig, axs = plt.subplots(rows, cols, figsize=figsize)

    # Ensure axs is iterable
    if isinstance(axs, plt.Axes):
        axs = np.array([axs])
    axs = axs.flatten()

    for fold_idx, cm in enumerate(cm_folds_norm):
        ax = axs[fold_idx]
        sns.heatmap(cm, annot=annots[fold_idx], fmt="", ax=ax, cbar=cbar,
                    vmin=0.0, vmax=1.0, cmap="Blues", linecolor='black',
                    xticklabels=class_names if class_names else "auto",
                    yticklabels=class_names if class_names else "auto")
        ax.set_title(f"Confusion Matrix - Fold {i_folds[fold_idx]}")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        # Add outer border
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(1)
            spine.set_edgecolor('black')

    # Remove unused subplots
    for j in range(k_fold, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, '_fold_all.png')
        plt.savefig(save_path)

    if show:
        plt.show()

    if save_dir:
        print("Saved in:", save_dir)

def save_metrics_to_json(metrics, save_dir, filename):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    # print("Saved as:", save_path)

def find_substr(str_arr, substr):
    return torch.tensor(np.char.find(str_arr, substr) != -1).bool()

def plot_anomaly_detection(dataset_person, dataset_study, outlier_thresh=(None, None)):
    mask = find_substr(dataset_person.ids, dataset_study)
    X = dataset_person.X[mask]
    y = dataset_person.y[mask]
    ids = dataset_person.ids[mask]

    N, W, F = X.shape
    X = X.reshape(N, W*F).numpy()
    X_list = [X[i, :] for i in range(X.shape[0])]
    X_df = pd.DataFrame({'X': X_list, 'y': y, 'id': ids})

    pca = PCA(n_components=2)
    X_df[['X_pca_0', 'X_pca_1']] = pca.fit_transform(X)

    X_df['outlier'] = 0
    if outlier_thresh[0] is not None:
        cond_lt = X_df['X_pca_1'] < outlier_thresh[0]
        cond_gt = X_df['X_pca_1'] > outlier_thresh[0]
        cond = cond_lt if cond_gt.sum() > cond_lt.sum() else cond_gt
        X_df.loc[cond, 'outlier'] = 1
    if outlier_thresh[1] is not None:
        cond_lt = X_df['X_pca_0'] < outlier_thresh[1]
        cond_gt = X_df['X_pca_0'] > outlier_thresh[1]
        cond = cond_lt if cond_gt.sum() > cond_lt.sum() else cond_gt
        X_df.loc[cond, 'outlier'] = 1

    plt.figure(figsize=(6, 4))

    for outlier, group in X_df.groupby('outlier'):
        plt.scatter(group['X_pca_1'], group['X_pca_0'], s=5, 
                    c='red' if outlier else 'blue', label='Anomaly' if outlier else 'Normal')

    plt.title(f"Anomaly Detection - {dataset_study}")
    plt.xlabel("PC-1")
    plt.ylabel("PC-2")

    if outlier_thresh[0] is not None:
        plt.axvline(x=outlier_thresh[0], color='red', linestyle=':')
    if outlier_thresh[1] is not None:
        plt.axhline(y=outlier_thresh[1], color='red', linestyle=':')
    
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_anomaly_detection_GaJuSi(dataset_person, outlier_thresh_map, s=10, figsize=(6, 4), save_dir='anomaly_detection'):
    dataset_studies = ['Ga', 'Ju', 'Si']

    n_cols = len(dataset_studies)
    n_rows = 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0] * n_cols, figsize[1] * n_rows))
    axes = axes.flatten()  # Flatten for easier indexing

    for i, dataset_study in enumerate(dataset_studies):
        outlier_thresh = outlier_thresh_map[dataset_study]

        mask = find_substr(dataset_person.ids, dataset_study)
        X = dataset_person.X[mask]
        y = dataset_person.y[mask]
        ids = dataset_person.ids[mask]

        N, W, F = X.shape
        X = X.reshape(N, W*F).numpy()
        X_list = [X[i, :] for i in range(X.shape[0])]
        X_df = pd.DataFrame({'X': X_list, 'y': y, 'id': ids})

        pca = PCA(n_components=2)
        X_df[['X_pca_0', 'X_pca_1']] = pca.fit_transform(X)

        X_df['outlier'] = 0
        if outlier_thresh[0] is not None:
            cond_lt = X_df['X_pca_1'] < outlier_thresh[0]
            cond_gt = X_df['X_pca_1'] > outlier_thresh[0]
            cond = cond_lt if cond_gt.sum() > cond_lt.sum() else cond_gt
            X_df.loc[cond, 'outlier'] = 1
        if outlier_thresh[1] is not None:
            cond_lt = X_df['X_pca_0'] < outlier_thresh[1]
            cond_gt = X_df['X_pca_0'] > outlier_thresh[1]
            cond = cond_lt if cond_gt.sum() > cond_lt.sum() else cond_gt
            X_df.loc[cond, 'outlier'] = 1
        
        print(f"Dataset - {dataset_study} anomaly data IDs:", X_df[X_df['outlier'] == 1]['id'].tolist())

        ax = axes[i]
        for outlier, group in X_df.groupby('outlier'):
            ax.scatter(group['X_pca_1'], group['X_pca_0'], s=s, 
                        c='red' if outlier else 'blue', label='Anomaly' if outlier else 'Normal')
        
        # Ensure both legends are shown
        unique_outliers = X_df['outlier'].unique()
        if 0 not in unique_outliers:
            ax.scatter([], [], s=s, c='blue', label='Normal')
        if 1 not in unique_outliers:
            ax.scatter([], [], s=s, c='red', label='Anomaly')

        ax.set_title(f"Anomaly Detection - {dataset_study}")
        ax.set_xlabel("PC-1")
        ax.set_ylabel("PC-2")

        if outlier_thresh[0] is not None:
            ax.axvline(x=outlier_thresh[0], color='red', linestyle=':')
        if outlier_thresh[1] is not None:
            ax.axhline(y=outlier_thresh[1], color='red', linestyle=':')

        ax.legend()

    plt.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, '_GaJuSi.png')
        plt.savefig(save_path)

    plt.show()

    if save_dir:
        print("Saved in:", save_dir)

def get_unique_rand_pair_idxs(n, limit):
    # Generate all possible pairs
    all_pairs = [(i, j) for i in range(limit) for j in range(limit) if i != j]

    # Shuffle the pairs
    random.shuffle(all_pairs)

    # Take the first n unique pairs
    unique_pairs = all_pairs[:n]

    return unique_pairs

def linear_interpolation(series1, series2, alpha=0.5):
    return alpha * series1 + (1 - alpha) * series2

def resample_linear_interpolation(X, n):
    X_new_shape = [0] + list(X.shape[1:])
    X_new = torch.empty(X_new_shape, dtype=torch.float32)
    
    for idx1, idx2 in get_unique_rand_pair_idxs(n, len(X)):
        X_1 = X[idx1]
        X_2 = X[idx2]
        
        X_new_i = linear_interpolation(X_1, X_2).unsqueeze(0)
        X_new = torch.cat((X_new, X_new_i), dim=0)
        
    return X_new

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')  # For Apple Silicon
    else:
        return torch.device('cpu')

def init_model(model_name, device, c_in, c_out, seq_len, bidirectional, layers=None, ps=None):
    if model_name == 'InceptionTime':
        return InceptionTime(c_in=c_in, c_out=c_out, seq_len=seq_len).to(device)
    elif model_name == 'RNN':
        return RNN(c_in=c_in, c_out=c_out, bidirectional=bidirectional).to(device)
    elif model_name == 'InceptionTimeRNN':
        return InceptionTimeRNN(c_in=c_in, c_out=c_out, seq_len=seq_len, bidirectional=bidirectional).to(device)
    elif model_name == 'MLP':
        return MLP(c_in=c_in, c_out=c_out, seq_len=seq_len, layers=layers, ps=ps).to(device)
    else:   # RNN-InceptionTime
        return RNNInceptionTime(c_in=c_in, c_out=c_out, seq_len=seq_len, bidirectional=bidirectional).to(device)