import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from typing import Tuple
from IPython import display
from torch.utils.data import Dataset, DataLoader
from scipy import stats as st
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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
    print(f"Dataset size (X, y): {len(X)}, {len(y)}")
    print("Label counts:")
    print(pd.DataFrame(y).value_counts().sort_index())

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

# def eval_person_majority_voting(model, dataset, window_size, stride_size, zeros_filter_thres=1.0, criterion=None, average='weighted', debug=False):
def eval_person_majority_voting(model, dataset, window_size, zeros_filter_thres=1.0, criterion=None, average='weighted', debug=False):
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
import random
import transformers

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

    print(f"Random seed set to: {seed}")