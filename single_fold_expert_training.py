import os
import fire
import numpy as np
import torch
from datetime import datetime
from torch.utils.data import TensorDataset, DataLoader

from src.utils import (
    set_seed, get_device, print_h, init_model,
    eval_window, eval_person_severity_voting, eval_person_majority_voting, eval_person_max_severity,
    init_metrics, update_metrics, save_metrics_to_json,
)

def main(
    fold_i_dir,
    model_name, # 'InceptionTime' | 'RNN' | 'InceptionTimeRNN' | 'RNNInceptionTime'
    bidirectional,
    n_epoch,
    seed = 69,

    # Training parameters
    batch_size = 8,
    n_feat = 16,
    n_class = 4,
    window_size = 500,
    lr = 3e-4,
):
    # Set seed and device
    set_seed(seed)
    device = get_device()
    print("Device:", device)

    # Generate name tag
    i_fold = int(fold_i_dir.split('fold_')[-1])
    print("Fold number:", i_fold)
    run_name_tag = fold_i_dir.split('/')[-2].rsplit('_v', 1)[0] + f'_fold_{i_fold:02}_e{n_epoch}'
    print("Run name tag:", run_name_tag)

    # Set run name
    run_name = f'{model_name}{'_bidirectional' if bidirectional else ''}_{run_name_tag+'_' if run_name_tag else ''}v{datetime.now().strftime("%Y%m%d%H%M%S")}'
    print("Run name:", run_name)

    # Create save directory
    save_dir = 'checkpoints/' + run_name
    os.makedirs(save_dir, exist_ok=True)
    print("Save directory:", save_dir)
    print()

    # Initialize evaluation metrics
    metrics = {
        'window': init_metrics(['acc', 'f1', 'precision', 'recall', 'cm', 'train_loss', 'val_loss']),
        'person_majority_voting': init_metrics(['acc', 'f1', 'precision', 'recall', 'cm', 'train_loss', 'val_loss']),
        'person_severity_voting': init_metrics(['acc', 'f1', 'precision', 'recall', 'cm', 'train_loss', 'val_loss']),
        'person_max_severity': init_metrics(['acc', 'f1', 'precision', 'recall', 'cm', 'train_loss', 'val_loss']),
    }

    # ================================================================================================================================
    # FOLD
    # ================================================================================================================================
    print_h(f"FOLD {i_fold}", 128)

    # ================================================================================================
    # DATA
    # ================================================================================================
    X_train_window = torch.tensor(np.load(os.path.join(fold_i_dir, f'X_train_window.npy'))).float()
    y_train_window = torch.tensor(np.load(os.path.join(fold_i_dir, f'y_train_window.npy'))).long()

    X_val_window = torch.tensor(np.load(os.path.join(fold_i_dir, f'X_val_window.npy'))).float()
    y_val_window = torch.tensor(np.load(os.path.join(fold_i_dir, f'y_val_window.npy'))).long()

    X_test_window = torch.tensor(np.load(os.path.join(fold_i_dir, f'X_test_window.npy'))).float()
    y_test_window = torch.tensor(np.load(os.path.join(fold_i_dir, f'y_test_window.npy'))).long()

    X_val_person = torch.tensor(np.load(os.path.join(fold_i_dir, f'X_val_person.npy'))).float()
    y_val_person = torch.tensor(np.load(os.path.join(fold_i_dir, f'y_val_person.npy'))).long()

    X_test_person = torch.tensor(np.load(os.path.join(fold_i_dir, f'X_test_person.npy'))).float()
    y_test_person = torch.tensor(np.load(os.path.join(fold_i_dir, f'y_test_person.npy'))).long()

    train_window_dataset = TensorDataset(X_train_window, y_train_window)
    val_window_dataset = TensorDataset(X_val_window, y_val_window)
    test_window_dataset = TensorDataset(X_test_window, y_test_window)

    val_person_dataset = TensorDataset(X_val_person, y_val_person)
    test_person_dataset = TensorDataset(X_test_person, y_test_person)

    train_dataloader = DataLoader(train_window_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_window_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_window_dataset, batch_size=batch_size, shuffle=False)

    # ================================================================================================
    # TRAINING
    # ================================================================================================
    print_h("TRAINING", 96)

    # Initialize model
    model = init_model(model_name, device, c_in=n_feat, c_out=n_class, seq_len=window_size, bidirectional=bidirectional)

    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion =  torch.nn.CrossEntropyLoss()

    # Swith the model to training mode
    model.train()

    # Loop training epochs
    global_val_loss_window_list = []
    global_val_loss_person_list = []
    global_train_loss_list = []
    train_loss_list = []
    # step = 0
    for epoch in range(n_epoch):
        # Loop training batches
        for iter, (X_train, y_train) in enumerate(train_dataloader):
            # Flush the computed gradients
            optimizer.zero_grad()
            
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            
            # Feed forward the model
            X_train = X_train.permute(0, 2, 1)
            y_pred = model(X_train)

            # Compute training loss
            train_loss = criterion(y_pred, y_train)
            train_loss_list.append(train_loss)
            
            # if (iter+1) % 'step_siz']= 0:
            if iter+1 == len(train_dataloader):
                # ================================================================
                # VALIDATION
                # ================================================================
                avg_val_loss_window, acc_window, f1_window, *_ = eval_window(model, val_dataloader, criterion, average='weighted')
                avg_val_loss_person, acc_person, f1_person, *_ = eval_person_majority_voting(model, test_person_dataset, criterion=criterion, average='weighted',
                                                                                                window_size=window_size)
                
                global_val_loss_window_list.append(avg_val_loss_window)
                global_val_loss_person_list.append(avg_val_loss_person)
                
                # Compute the average training loss for each epoch
                avg_train_loss = sum(train_loss_list) / len(train_dataloader)
                global_train_loss_list.append(avg_train_loss.item())
                train_loss_list = []
                
                # ================================================================
                # LOGGING
                # ================================================================
                print(f"epoch: {epoch+1}, "
                    # f"iter: {iter+1}, "
                    # f"step: {step+1}, "
                    f"train/loss: {avg_train_loss:.3f}, "
                    f"val/loss_window: {avg_val_loss_window:.3f}, "
                    f"val/acc_window: {acc_window:.3f}, "
                    f"val/f1_window: {f1_window:.3f}, "
                    f"val/loss_person: {avg_val_loss_person:.3f}, "
                    f"val/acc_person: {acc_person:.3f}, "
                    f"val/f1_person: {f1_person:.3f}"
                )
                
                # Switch the model back to training mode
                model.train()
                
                # step += 1
            
            # Backward pass the model
            train_loss.backward()
            
            # Update the model weights based on computed gradients
            optimizer.step()
    print()

    # ================================================================================================
    # EVALUATION
    # ================================================================================================
    print_h("EVALUATION", 96)

    # ================================================================
    # EVALUATION ON WINDOW DATA
    # ================================================================
    print_h("EVALUATION ON WINDOW DATA", 64)
    (
        _, 
        acc_window, 
        f1_window, 
        precision_window, 
        recall_window, 
        cm_window
    ) = eval_window(
        model, 
        test_dataloader, 
        average='weighted',
    )
    print("acc:", acc_window)
    print("f1:", f1_window)
    print("precision:", precision_window)
    print("recall:", recall_window)
    print("cm:\n", np.array(cm_window))
    print()

    # ================================================================
    # EVALUATION ON PERSON DATA BY MAJORITY VOTING
    # ================================================================
    print_h("EVALUATION ON PERSON DATA BY MAJORITY VOTING", 64)
    (
        _,
        acc_person_majority_voting,
        f1_person_majority_voting,
        precision_person_majority_voting,
        recall_person_majority_voting,
        cm_person_majority_voting,
        *_,
    ) = eval_person_majority_voting(
        model, 
        test_person_dataset, 
        criterion=None, 
        average='weighted',
        window_size=window_size, 
        debug=False,
        seed=seed,
    )
    print("acc:", acc_person_majority_voting)
    print("f1:", f1_person_majority_voting)
    print("precision:", precision_person_majority_voting)
    print("recall:", recall_person_majority_voting)
    print("cm:\n", np.array(cm_person_majority_voting))
    print()

    # ================================================================
    # EVALUATION ON PERSON DATA BY SEVERITY VOTING
    # ================================================================
    print_h("EVALUATION ON PERSON DATA BY SEVERITY VOTING", 64)
    (
        _, 
        acc_person_severity_voting, 
        f1_person_severity_voting, 
        precision_person_severity_voting, 
        recall_person_severity_voting, 
        cm_person_severity_voting,
    ) = eval_person_severity_voting(
        model, 
        test_person_dataset, 
        criterion=None, 
        average='weighted',
        window_size=window_size, 
        debug=False,
        seed=seed,
    )
    print("acc:", acc_person_severity_voting)
    print("f1:", f1_person_severity_voting)
    print("precision:", precision_person_severity_voting)
    print("recall:", recall_person_severity_voting)
    print("cm:\n", np.array(cm_person_severity_voting))
    print()

    # ================================================================
    # EVALUATION ON PERSON DATA BY MAX. SEVERITY
    # ================================================================
    print_h("EVALUATION ON PERSON DATA BY MAX. SEVERITY", 64)
    (
        _, 
        acc_person_max_severity, 
        f1_person_max_severity, 
        precision_person_max_severity, 
        recall_person_max_severity, 
        cm_person_max_severity,
    ) = eval_person_max_severity(
        model, 
        test_person_dataset, 
        criterion=None, 
        average='weighted',
        window_size=window_size, 
        debug=False,
        seed=seed,
    )
    print("acc:", acc_person_max_severity)
    print("f1:", f1_person_max_severity)
    print("precision:", precision_person_max_severity)
    print("recall:", recall_person_max_severity)
    print("cm:\n", np.array(cm_person_max_severity))
    print()

    in_metrics = {
        'person_majority_voting': {
            'acc': acc_person_majority_voting,
            'f1': f1_person_majority_voting,
            'precision': precision_person_majority_voting,
            'recall': recall_person_majority_voting,
            'cm': cm_person_majority_voting,
        },
        'person_severity_voting': {
            'acc': acc_person_severity_voting,
            'f1': f1_person_severity_voting,
            'precision': precision_person_severity_voting,
            'recall': recall_person_severity_voting,
            'cm': cm_person_severity_voting,
        },
        'person_max_severity': {
            'acc': acc_person_max_severity,
            'f1': f1_person_max_severity,
            'precision': precision_person_max_severity,
            'recall': recall_person_max_severity,
            'cm': cm_person_max_severity,
        },
        'window': {
            'acc': acc_window,
            'f1': f1_window,
            'precision': precision_window,
            'recall': recall_window,
            'cm': cm_window,
        },
    }

    for metric_type in in_metrics.keys():
        update_metrics(metrics[metric_type], in_metrics[metric_type])

    metrics['window']['train_loss']['folds'].append(global_train_loss_list)
    metrics['window']['val_loss']['folds'].append(global_val_loss_window_list)

    # ================================================================================================
    # CHECKPOINT SAVING
    # ================================================================================================
    save_path = os.path.join(save_dir, f'fold_{i_fold:02}.pth')
    torch.save(model.state_dict(), save_path)
    print(f"Checkpoint for fold {i_fold:02} is saved to:", save_path)

    # DEBUG: Test for only one fold
    # break

    save_metrics_to_json(metrics, save_dir, filename='_evaluation_metrics.json')
    print("Evaluation metrics is saved in:", save_dir)

if __name__ == '__main__':
    fire.Fire(main)