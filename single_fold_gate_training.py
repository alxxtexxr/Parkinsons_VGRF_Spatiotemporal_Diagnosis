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
from src.models import HardMoE

def main(
    gate_name, # 'InceptionTime' | 'RNN' | 'InceptionTimeRNN' | 'RNNInceptionTime' | 'MLP'
    expert_model_path_Ga,
    expert_model_path_Ju,
    expert_model_path_Si,
    fold_i_dir_Ga,
    fold_i_dir_Ju,
    fold_i_dir_Si,
    n_epoch,
    seed = 69,

    # Model parameters
    gate_bidirectional = True,
    gate_layers = None,
    gate_ps = None,

    # Training parameters
    batch_size = 8,
    n_feat = 16,
    n_class = 4,
    window_size = 500,
    max_vgrf_data_len = 25_000,
    lr = 3e-4,
):
    # Set seed and device
    set_seed(seed)
    device = get_device()
    print("Device:", device)

    # Set up expert model checkpoint file path mapping
    expert_model_path_map = {
        'Ga': expert_model_path_Ga,
        'Ju': expert_model_path_Ju,
        'Si': expert_model_path_Si,
    }

    # Get model names and bidirectional parameter
    expert_names = [expert_model_dir.split('/')[-2].split('_'+study)[0] for study, expert_model_dir in expert_model_path_map.items()]
    assert len(set(expert_names)) == 1, f"Expert model names are inconsistent: {expert_names}"
    expert_name = expert_names[0]
    if 'bidirectional' in expert_name:
        expert_name = expert_name.replace('_bidirectional', '')
        expert_bidirectional = True
    else:
        expert_bidirectional = False
    print("Expert model name:", expert_name)
    print("Expert model bidirectional:", expert_bidirectional)

    # Set up single fold data directory mapping
    fold_i_dir_map = {
        'Ga': fold_i_dir_Ga,
        'Ju': fold_i_dir_Ju,
        'Si': fold_i_dir_Si,
    } 

    # Get fold number
    i_folds_data = [int(fold_i_dir.split('fold_')[-1]) for fold_i_dir in fold_i_dir_map.values()]
    i_folds_checkpoints = [int(expert_model_path.split('fold_')[-1].replace('.pth', '')) for expert_model_path in expert_model_path_map.values()]
    i_folds = i_folds_data + i_folds_checkpoints
    assert len(set(i_folds)) == 1, f"Fold numbers are inconsistent: {({'data': i_folds_data, 'checkpoints': i_folds_checkpoints})}"
    i_fold = i_folds[0]
    print("Fold number:", i_fold)

    # Generate name tags
    run_name_tag = '_'.join([fold_i_dir.split('/')[-2].rsplit('_v', 1)[0] for fold_i_dir in fold_i_dir_map.values()]) + f'_fold_{i_fold:02}_e{n_epoch}'
    gate_name_tag = f'{gate_name}Gate{'_bidirectional' if gate_bidirectional and gate_name in ['RNNInceptionTime', 'RNN'] else ''}{f'_l{len(gate_layers)}' if gate_layers and gate_name in ['MLP'] else ''}'
    moe_name_tag = f'{expert_name}MoE{'_bidirectional' if expert_bidirectional else ''}{f'_{gate_name_tag}' if gate_name != expert_name else ''}'
    print("Run name tag:", run_name_tag)
    print("Gate model name tag:", gate_name_tag)
    print("MoE model name tag:", moe_name_tag)

    # Set run names
    v = datetime.now().strftime("%Y%m%d%H%M%S")
    gate_run_name = f'{gate_name_tag}_{run_name_tag+'_' if run_name_tag else ''}v{v}'
    moe_run_name = f'{moe_name_tag}_{run_name_tag+'_' if run_name_tag else ''}v{v}'
    print("Gate model run name:", gate_run_name)
    print("MoE model run name:", moe_run_name)
    print()

    # Create save directories
    gate_save_dir = 'checkpoints/' + gate_run_name
    moe_save_dir = 'checkpoints/' + moe_run_name
    os.makedirs(gate_save_dir, exist_ok=True)
    os.makedirs(moe_save_dir, exist_ok=True)
    print("Gate model save directory:", gate_save_dir)
    print("MoE model save directory:", moe_save_dir)
    print()

    # Initialize evaluation metrics
    gate_metrics = {
        'window': init_metrics(['acc', 'f1', 'precision', 'recall', 'cm', 'train_loss', 'val_loss']),
        # 'person_majority_voting': init_metrics(['acc', 'f1', 'precision', 'recall', 'cm', 'train_loss', 'val_loss']),
        # 'person_severity_voting': init_metrics(['acc', 'f1', 'precision', 'recall', 'cm', 'train_loss', 'val_loss']),
        # 'person_max_severity': init_metrics(['acc', 'f1', 'precision', 'recall', 'cm', 'train_loss', 'val_loss']),
    }
    moe_metrics = {
        # 'window': init_metrics(['acc', 'f1', 'precision', 'recall', 'cm']),
        'person_majority_voting': init_metrics(['acc', 'f1', 'precision', 'recall', 'cm']),
        # 'person_severity_voting': init_metrics(['acc', 'f1', 'precision', 'recall', 'cm']),
        # 'person_max_severity': init_metrics(['acc', 'f1', 'precision', 'recall', 'cm']),
    }

    study_label_map = {
        'Ga': 0,
        'Ju': 1,
        'Si': 2,
    }

    # ================================================================================================================================
    # FOLD
    # ================================================================================================================================
    print_h(f"FOLD {i_fold}", 128)

    expert_model_map = {
        'Ga': init_model(expert_name, device, c_in=n_feat, c_out=n_class, seq_len=window_size, bidirectional=expert_bidirectional),
        'Ju': init_model(expert_name, device, c_in=n_feat, c_out=n_class, seq_len=window_size, bidirectional=expert_bidirectional),
        'Si': init_model(expert_name, device, c_in=n_feat, c_out=n_class, seq_len=window_size, bidirectional=expert_bidirectional),
    }

    X_train_window_GaJuSi = torch.empty(0, window_size, n_feat).float()
    y_train_window_GaJuSi = torch.empty(0).long()
    study_labels_train_window_GaJuSi = torch.empty(0).long()

    X_val_window_GaJuSi = torch.empty(0, window_size, n_feat).float()
    y_val_window_GaJuSi = torch.empty(0).long()
    study_labels_val_window_GaJuSi = torch.empty(0).long()

    X_test_window_GaJuSi = torch.empty(0, window_size, n_feat).float()
    y_test_window_GaJuSi = torch.empty(0).long()
    study_labels_test_window_GaJuSi = torch.empty(0).long()

    X_val_person_GaJuSi = torch.empty(0, max_vgrf_data_len, n_feat).float()
    y_val_person_GaJuSi = torch.empty(0).long()
    # study_labels_val_person_GaJuSi = torch.empty(0).long()

    X_test_person_GaJuSi = torch.empty(0, max_vgrf_data_len, n_feat).float()
    y_test_person_GaJuSi = torch.empty(0).long()
    # study_labels_test_person_GaJuSi = torch.empty(0).long()

    for study, fold_i_dir in fold_i_dir_map.items():
        # ================================================================================================
        # EXPERT MODEL
        # ================================================================================================
        print_h(f"EXPERT-{study} MODEL", 96)

        X_train_window = torch.tensor(np.load(os.path.join(fold_i_dir, f'X_train_window.npy'))).float()
        y_train_window = torch.tensor(np.load(os.path.join(fold_i_dir, f'y_train_window.npy'))).long()
        study_labels_train_window = torch.tensor([study_label_map[study]] * len(y_train_window)).long()
        X_train_window_GaJuSi = torch.cat((X_train_window_GaJuSi, X_train_window), dim=0)
        y_train_window_GaJuSi = torch.cat((y_train_window_GaJuSi, y_train_window), dim=0)
        study_labels_train_window_GaJuSi = torch.cat((study_labels_train_window_GaJuSi, study_labels_train_window), dim=0)

        X_val_window = torch.tensor(np.load(os.path.join(fold_i_dir, f'X_val_window.npy'))).float()
        y_val_window = torch.tensor(np.load(os.path.join(fold_i_dir, f'y_val_window.npy'))).long()
        study_labels_val_window = torch.tensor([study_label_map[study]] * len(y_val_window)).long()
        X_val_window_GaJuSi = torch.cat((X_val_window_GaJuSi, X_val_window), dim=0)
        y_val_window_GaJuSi = torch.cat((y_val_window_GaJuSi, y_val_window), dim=0)
        study_labels_val_window_GaJuSi = torch.cat((study_labels_val_window_GaJuSi, study_labels_val_window), dim=0)

        X_test_window = torch.tensor(np.load(os.path.join(fold_i_dir, f'X_test_window.npy'))).float()
        y_test_window = torch.tensor(np.load(os.path.join(fold_i_dir, f'y_test_window.npy'))).long()
        study_labels_test_window = torch.tensor([study_label_map[study]] * len(y_test_window)).long()
        X_test_window_GaJuSi = torch.cat((X_test_window_GaJuSi, X_test_window), dim=0)
        y_test_window_GaJuSi = torch.cat((y_test_window_GaJuSi, y_test_window), dim=0)
        study_labels_test_window_GaJuSi = torch.cat((study_labels_test_window_GaJuSi, study_labels_test_window), dim=0)

        X_val_person = torch.tensor(np.load(os.path.join(fold_i_dir, f'X_val_person.npy'))).float()
        y_val_person = torch.tensor(np.load(os.path.join(fold_i_dir, f'y_val_person.npy'))).long()
        X_val_person_GaJuSi = torch.cat((X_val_person_GaJuSi, X_val_person), dim=0)
        y_val_person_GaJuSi = torch.cat((y_val_person_GaJuSi, y_val_person), dim=0)

        X_test_person = torch.tensor(np.load(os.path.join(fold_i_dir, f'X_test_person.npy'))).float()
        y_test_person = torch.tensor(np.load(os.path.join(fold_i_dir, f'y_test_person.npy'))).long()
        X_test_person_GaJuSi = torch.cat((X_test_person_GaJuSi, X_test_person), dim=0)
        y_test_person_GaJuSi = torch.cat((y_test_person_GaJuSi, y_test_person), dim=0)

        train_window_dataset = TensorDataset(X_train_window, y_train_window)
        val_window_dataset = TensorDataset(X_val_window, y_val_window)
        test_window_dataset = TensorDataset(X_test_window, y_test_window)
        
        val_person_dataset = TensorDataset(X_val_person, y_val_person)
        test_person_dataset = TensorDataset(X_test_person, y_test_person)

        train_dataloader = DataLoader(train_window_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_window_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_window_dataset, batch_size=batch_size, shuffle=False)
        
        expert_model = expert_model_map[study]

        # Load pretrained model
        model_i_path = expert_model_path_map[study]
        expert_model.load_state_dict(torch.load(model_i_path, map_location=device))

        # ================================================================
        # EXPERT MODEL EVALUATION ON PERSON DATA BY MAJORITY VOTING
        # ================================================================
        print_h("EVALUATION ON PERSON DATA BY MAJORITY VOTING", 64)
        _, acc_person_majority_voting, f1_person_majority_voting, precision_person_majority_voting, recall_person_majority_voting, cm_person_majority_voting, *_ = eval_person_majority_voting(expert_model, val_person_dataset, criterion=None, average='weighted',
                                                                                                                                                                                                window_size=window_size, debug=False)
        print("acc:", acc_person_majority_voting)
        print("f1:", f1_person_majority_voting)
        print("precision:", precision_person_majority_voting)
        print("recall:", recall_person_majority_voting)
        print("cm:\n", np.array(cm_person_majority_voting))
        print()

    # ================================================================================================
    # GATE MODEL
    # ================================================================================================
    print_h("GATE MODEL", 96)

    # train_window_dataset_GaJuSi = TensorDataset(X_train_window_GaJuSi, y_train_window_GaJuSi)
    # val_window_dataset_GaJuSi = TensorDataset(X_val_window_GaJuSi, y_val_window_GaJuSi)
    # test_window_dataset_GaJuSi = TensorDataset(X_test_window_GaJuSi, y_test_window_GaJuSi)

    train_window_dataset_GaJuSi = TensorDataset(X_train_window_GaJuSi, study_labels_train_window_GaJuSi)
    val_window_dataset_GaJuSi = TensorDataset(X_val_window_GaJuSi, study_labels_val_window_GaJuSi)
    test_window_dataset_GaJuSi = TensorDataset(X_test_window_GaJuSi, study_labels_test_window_GaJuSi)

    train_dataloader_GaJuSi = DataLoader(train_window_dataset_GaJuSi, batch_size=batch_size, shuffle=True)
    val_dataloader_GaJuSi = DataLoader(val_window_dataset_GaJuSi, batch_size=batch_size, shuffle=False)
    test_dataloader_GaJuSi = DataLoader(test_window_dataset_GaJuSi, batch_size=batch_size, shuffle=False)

    # ================================================================
    # GATE MODEL TRAINING
    # ================================================================
    print_h("TRAINING", 64)
    gate_model = init_model(gate_name, device, c_in=n_feat, c_out=len(study_label_map.keys()), seq_len=window_size, bidirectional=gate_bidirectional)

    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(gate_model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Swith the model to training mode
    gate_model.train()

    # Loop training epochs
    global_val_loss_window_list = []
    global_val_loss_person_list = []
    global_train_loss_list = []
    train_loss_list = []
    # step = 0
    for epoch in range(n_epoch):
        # Loop training batches
        for iter, (X_train, y_train) in enumerate(train_dataloader_GaJuSi):
            # Flush the computed gradients
            optimizer.zero_grad()
            
            X_train = X_train.to(device)
            y_train = y_train.to(device)
            
            # Feed forward the model
            X_train = X_train.permute(0, 2, 1)
            y_pred = gate_model(X_train)

            # print(f'{X_train=}')
            # print(f'{y_pred=}')
            
            # Compute training loss
            train_loss = criterion(y_pred, y_train)
            train_loss_list.append(train_loss)
            
            # if (iter+1) % 'step_siz']= 0:
            if iter+1 == len(train_dataloader_GaJuSi):
                # ================================
                # GATE MODEL VALIDATION
                # ================================
                avg_val_loss_window, acc_window, f1_window, _, _, _ = eval_window(gate_model, val_dataloader_GaJuSi, criterion, average='weighted')
                # avg_val_loss_person, acc_person, f1_person, _, _, _ = eval_person_majority_voting(model, test_person_dataset_GaJuSi, criterion=criterion, average='weighted',
                #                                                                                   window_size=window_size, zeros_filter_thres=zeros_filter_thres)
                
                global_val_loss_window_list.append(avg_val_loss_window)
                # global_val_loss_person_list.append(avg_val_loss_person)
                
                # Compute the average training loss for each epoch
                avg_train_loss = sum(train_loss_list) / len(train_dataloader)
                global_train_loss_list.append(avg_train_loss.item())
                train_loss_list = []
                
                # ================================
                # GATE MODEL LOGGING
                # ================================
                print(f"epoch: {epoch+1}, "
                    # f"iter: {iter+1}, "
                    # f"step: {step+1}, "
                    f"train/loss: {avg_train_loss:.3f}, "
                    f"val/loss_window: {avg_val_loss_window:.3f}, "
                    f"val/acc_window: {acc_window:.3f}, "
                    f"val/f1_window: {f1_window:.3f}, "
                    # f"val/loss_person: {avg_val_loss_person:.3f}, "
                    # f"val/acc_person: {acc_person:.3f}, "
                    # f"val/f1_person: {f1_person:.3f}"
                )
                
                # Switch the model back to training mode
                gate_model.train()
                
                # step += 1
            
            # Backward pass the model
            train_loss.backward()
            
            # Update the model weights based on computed gradients
            optimizer.step()
    print()

    # ================================================================
    # GATE MODEL EVALUATION ON WINDOW DATA
    # ================================================================
    print_h("EVALUATION ON WINDOW DATA", 64)

    _, acc_window, f1_window, precision_window, recall_window, cm_window = eval_window(gate_model, test_dataloader_GaJuSi, average='weighted')

    print("acc:", acc_window)
    print("f1:", f1_window)
    print("precision:", precision_window)
    print("recall:", recall_window)
    print("cm:\n", np.array(cm_window))
    print()

    gate_in_metrics = {
            'window': {
            'acc': acc_window,
            'f1': f1_window,
            'precision': precision_window,
            'recall': recall_window,
            'cm': cm_window,
        },
        # 'person_majority_voting': {
        #     'acc': acc_person_majority_voting,
        #     'f1': f1_person_majority_voting,
        #     'precision': precision_person_majority_voting,
        #     'recall': recall_person_majority_voting,
        #     'cm': cm_person_majority_voting,
        # },
        # 'person_severity_voting': {
        #     'acc': acc_person_severity_voting,
        #     'f1': f1_person_severity_voting,
        #     'precision': precision_person_severity_voting,
        #     'recall': recall_person_severity_voting,
        #     'cm': cm_person_severity_voting,
        # },
        # 'person_max_severity': {
        #     'acc': acc_person_max_severity,
        #     'f1': f1_person_max_severity,
        #     'precision': precision_person_max_severity,
        #     'recall': recall_person_max_severity,
        #     'cm': cm_person_max_severity,
        # },   
    }

    for metric_type in gate_in_metrics.keys():
        update_metrics(gate_metrics[metric_type], gate_in_metrics[metric_type])

    gate_metrics['window']['train_loss']['folds'].append(global_train_loss_list)
    gate_metrics['window']['val_loss']['folds'].append(global_val_loss_window_list)

    # ================================================================================================
    # MoE MODEL
    # ================================================================================================
    print_h("MoE MODEL", 96)

    val_person_dataset_GaJuSi = TensorDataset(X_val_person_GaJuSi, y_val_person_GaJuSi)
    test_person_dataset_GaJuSi = TensorDataset(X_test_person_GaJuSi, y_test_person_GaJuSi)

    moe_model = HardMoE(experts=expert_model_map.values(), gate=gate_model)

    # ================================================================
    # MoE MODEL EVALUATION ON PERSON DATA BY MAJORITY VOTING
    # ================================================================
    print_h("EVALUATION ON PERSON DATA BY MAJORITY VOTING", 64)
    _, acc_person_majority_voting, f1_person_majority_voting, precision_person_majority_voting, recall_person_majority_voting, cm_person_majority_voting, *_ = eval_person_majority_voting(moe_model, val_person_dataset_GaJuSi, criterion=None, average='weighted',
                                                                                                                                                                                        window_size=window_size, debug=False)
    print("acc:", acc_person_majority_voting)
    print("f1:", f1_person_majority_voting)
    print("precision:", precision_person_majority_voting)
    print("recall:", recall_person_majority_voting)
    print("cm:\n", np.array(cm_person_majority_voting))
    print()

    moe_in_metrics = {
            # 'window': {
        #     'acc': acc_window,
        #     'f1': f1_window,
        #     'precision': precision_window,
        #     'recall': recall_window,
        #     'cm': cm_window,
        # },
        'person_majority_voting': {
            'acc': acc_person_majority_voting,
            'f1': f1_person_majority_voting,
            'precision': precision_person_majority_voting,
            'recall': recall_person_majority_voting,
            'cm': cm_person_majority_voting,
        },
        # 'person_severity_voting': {
        #     'acc': acc_person_severity_voting,
        #     'f1': f1_person_severity_voting,
        #     'precision': precision_person_severity_voting,
        #     'recall': recall_person_severity_voting,
        #     'cm': cm_person_severity_voting,
        # },
        # 'person_max_severity': {
        #     'acc': acc_person_max_severity,
        #     'f1': f1_person_max_severity,
        #     'precision': precision_person_max_severity,
        #     'recall': recall_person_max_severity,
        #     'cm': cm_person_max_severity,
        # },   
    }

    for metric_type in moe_in_metrics.keys():
        update_metrics(moe_metrics[metric_type], moe_in_metrics[metric_type])

    # ================================================================================================
    # GATE MODEL CHECKPOINT SAVING
    # ================================================================================================
    gate_save_path = os.path.join(gate_save_dir, f'fold_{(i_fold):02}.pth')
    torch.save(gate_model.state_dict(), gate_save_path)

    print(f"Gate model checkpoint for fold {i_fold} is saved to:", gate_save_path)

    # ================================================================================================
    # MoE MODEL SAVING
    # ================================================================================================
    moe_save_path = os.path.join(moe_save_dir, f'fold_{(i_fold):02}.pth')
    torch.save(moe_model.state_dict(), moe_save_path)

    print(f"MoE model checkpoint for fold {i_fold} is saved to:", moe_save_path)
    print()

    # DEBUG: Test for only 1 fold
    # break

    save_metrics_to_json(moe_metrics, moe_save_dir, filename='_evaluation_metrics.json')
    print("MoE model evaluation metrics is saved in:", moe_save_dir)

    save_metrics_to_json(gate_metrics, gate_save_dir, filename='_evaluation_metrics.json')
    print("Gate model evaluation metrics is saved in:", gate_save_dir)

if __name__ == '__main__':
    fire.Fire(main)