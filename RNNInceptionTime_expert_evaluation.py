import os
import fire
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from pprint import pprint

from src.utils import (
    # Old utils
    print_h, eval_person_majority_voting,

    # New utils
    set_seed, get_device, init_metrics, update_metrics, save_metrics_to_json,
)
from src.models import RNNInceptionTime

def main(
    k_fold_dir,
    model_dir,
    seed = 69,
    
    # Evaluation config
    batch_size = 8,
    n_feat = 16,
    n_class = 4,
    window_size = 500,
    max_vgrf_data_len = 25_000,
    evaluations_dir = 'evaluations',
):
    # Set seed and device
    set_seed(seed)
    device = get_device()
    print("Device:", device)

    # Get model name and dataset study
    model_name = model_dir.split('/')[-1]
    study = model_name.rsplit('_k')[0].rsplit('_')[-1]
    print("Dataset study:", study)
    
    # Initialize evaluation metrics
    general_metrics_dir = f'evaluations/{model_name}/_general_metrics'
    metrics = {
        'person_majority_voting': init_metrics(['acc', 'f1', 'precision', 'recall', 'cm']),
        # 'person_severity_voting': init_metrics(['acc', 'f1', 'precision', 'recall', 'cm']),
        # 'person_max_severity': init_metrics(['acc', 'f1', 'precision', 'recall', 'cm']),
        # 'window': init_metrics(['acc', 'f1', 'precision', 'recall', 'cm']),
    }

    study_label_map = {
        'Ga': 0,
        'Ju': 1,
        'Si': 2,
    }

    for fold_i_dir_name in sorted(os.listdir(k_fold_dir)):
        # ================================================================================================================================
        # FOLD
        # ================================================================================================================================
        fold_i_dir = os.path.join(k_fold_dir, fold_i_dir_name)
        print_h(fold_i_dir_name, 128)

        # ================================================================================================
        # DATA
        # ================================================================================================
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

        # ================================================================================================s
        # MODEL
        # ================================================================================================s
        print_h(f"EXPERT-{study} MODEL", 96)
        
        # Load pretrained expert model
        model = RNNInceptionTime(c_in=n_feat, c_out=n_class, seq_len=window_size, bidirectional=True).to(device)
        model_i_path = os.path.join(model_dir, fold_i_dir_name + '.pth')
        model.load_state_dict(torch.load(model_i_path, map_location=device))

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
            *_
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

        # metrics = update_metrics(metrics, {
        #     'acc': acc_person_majority_voting,
        #     'f1': f1_person_majority_voting,
        #     'precision': precision_person_majority_voting,
        #     'recall': recall_person_majority_voting,
        #     'cm': cm_person_majority_voting,
        # })

        in_metrics = {
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
            # 'window': {
            #     'acc': acc_window,
            #     'f1': f1_window,
            #     'precision': precision_window,
            #     'recall': recall_window,
            #     'cm': cm_window,
            # },
        }

        for metric_type in in_metrics.keys():
            update_metrics(metrics[metric_type], in_metrics[metric_type])

        # DEBUG: Test for only one fold
        # break

    print_h("METRICS", 128)
    save_metrics_to_json(metrics, general_metrics_dir, f'_{study}.json')
    pprint(metrics, sort_dicts=False)
    print()
    print("Evaluation metrics is saved in:", general_metrics_dir)

if __name__ == '__main__':
    fire.Fire(main)