import os
import fire
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from pprint import pprint

from src.utils import (
    # Old utils
    print_h, eval_window, eval_person_majority_voting,

    # New utils
    set_seed, get_device, init_metrics, update_metrics, save_metrics_to_json,
    plot_k_fold_roc_curves_multiclass_v2, plot_k_fold_cm,
)
from src.models import RNNInceptionTime, HardMoE

def main(
    k_fold_dir_Ga,
    k_fold_dir_Ju,
    k_fold_dir_Si,
    expert_model_dir_Ga,
    expert_model_dir_Ju,
    expert_model_dir_Si,
    gate_model_dir,
    seed = 69,

    # Evaluation config
    k_fold = 10,
    batch_size = 8,
    n_feat = 16,
    n_class = 4,
    window_size = 500,
    max_vgrf_data_len = 25_000,
    evaluations_dir = 'evaluations',
):
    set_seed(seed)
    device = get_device()
    print("Device:", device)

    # Set up K-fold dataset and expert model mappings
    k_fold_dir_map = {
        'Ga': k_fold_dir_Ga,
        'Ju': k_fold_dir_Ju,
        'Si': k_fold_dir_Si,
    }
    expert_model_dir_map = {
        'Ga': expert_model_dir_Ga,
        'Ju': expert_model_dir_Ju,
        'Si': expert_model_dir_Si,
    }
    
    # Set up model path mappings and get number of folds (K-fold)
    expert_model_path_map = {study: sorted([model_dir_study+'/'+f for f in os.listdir(model_dir_study) if f.endswith('.pth')])
                            for study, model_dir_study in expert_model_dir_map.items()}
    gate_model_path_map = sorted([gate_model_dir+'/'+f for f in os.listdir(gate_model_dir) if f.endswith('.pth')])
    assert len(set([len(expert_model_path_study) for expert_model_path_study in expert_model_path_map.values()])) == 1, \
        f"Inconsistent number of folds across dataset studies: {[len(v) for v in expert_model_path_map.values()]}"
    k_fold = len(list(expert_model_path_map.values())[0])
    print("K-fold:", k_fold)

    general_metrics_dir = evaluations_dir + '/RNNInceptionTimeMoE_' + gate_model_dir.rsplit('RNNInceptionTime_')[-1].split('_v')[0] + '/_general_metrics'
    cm_dir = evaluations_dir + '/RNNInceptionTimeMoE_' + gate_model_dir.rsplit('RNNInceptionTime_')[-1].split('_v')[0] + '/cm'
    roc_curves_dir = evaluations_dir + '/RNNInceptionTimeMoE_' + gate_model_dir.rsplit('RNNInceptionTime_')[-1].split('_v')[0] + '/roc_curves'
    print("Evaluation general metrics save directory:", general_metrics_dir)
    print("Evaluation confusion matrix save directory:", cm_dir)
    print("Evaluation ROC curves save directory:", roc_curves_dir)

    moe_metrics = init_metrics(['acc', 'f1', 'precision', 'recall', 'cm', 
                                'fpr_multiclass_list', 'tpr_multiclass_list', 'roc_auc_multiclass_list', 'roc_auc_multiclass_avg'])
    gate_metrics = init_metrics(['acc', 'f1', 'precision', 'recall', 'cm'])
    expert_metrics = {
        'Ga': init_metrics(['acc', 'f1', 'precision', 'recall', 'cm']),
        'Ju': init_metrics(['acc', 'f1', 'precision', 'recall', 'cm']),
        'Si': init_metrics(['acc', 'f1', 'precision', 'recall', 'cm']),
    }

    study_label_map = {
        'Ga': 0,
        'Ju': 1,
        'Si': 2,
    }

    for i_fold in range(k_fold):
        print_h(f"FOLD {i_fold+1}", 128)
        
        expert_model_map = {
            'Ga': RNNInceptionTime(c_in=n_feat, c_out=n_class, seq_len=window_size, bidirectional=True).to(device),
            'Ju': RNNInceptionTime(c_in=n_feat, c_out=n_class, seq_len=window_size, bidirectional=True).to(device),
            'Si': RNNInceptionTime(c_in=n_feat, c_out=n_class, seq_len=window_size, bidirectional=True).to(device),
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

        for study, k_fold_dir in k_fold_dir_map.items():
            print_h(f"EXPERT-{study} MODEL", 96)

            fold_i_dir_name = os.listdir(k_fold_dir)[i_fold]
            fold_i_dir = os.path.join(k_fold_dir, fold_i_dir_name)

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

            # Load pretrained expert model
            expert_model_i_path = expert_model_path_map[study][i_fold]
            expert_model.load_state_dict(torch.load(expert_model_i_path, map_location=device))

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
                expert_model, 
                val_person_dataset, 
                criterion=None, 
                average='weighted',
                window_size=window_size, 
                debug=False
            )
            print("acc:", acc_person_majority_voting)
            print("f1:", f1_person_majority_voting)
            print("precision:", precision_person_majority_voting)
            print("recall:", recall_person_majority_voting)
            print("cm:\n", np.array(cm_person_majority_voting))
            print()

            expert_metrics[study] = update_metrics(expert_metrics[study], {
                'acc': acc_person_majority_voting,
                'f1': f1_person_majority_voting,
                'precision': precision_person_majority_voting,
                'recall': recall_person_majority_voting,
                'cm': cm_person_majority_voting,
            })

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

        gate_model = RNNInceptionTime(c_in=n_feat, c_out=len(study_label_map.keys()), seq_len=window_size, bidirectional=True).to(device)

        # Load pretrained gate model
        gate_model_i_path = gate_model_path_map[i_fold]
        gate_model.load_state_dict(torch.load(gate_model_i_path, map_location=device))

        print_h("EVALUATION ON WINDOW DATA", 64)
        
        _, acc_window, f1_window, precision_window, recall_window, cm_window = eval_window(gate_model, test_dataloader_GaJuSi, average='weighted')

        print("acc:", acc_window)
        print("f1:", f1_window)
        print("precision:", precision_window)
        print("recall:", recall_window)
        print("cm:\n", np.array(cm_window))
        print()

        gate_metrics = update_metrics(gate_metrics, {
            'acc': acc_window,
            'f1': f1_window,
            'precision': precision_window,
            'recall': recall_window,
            'cm': cm_window,
        })

        print_h("MoE MODEL", 96)

        val_person_dataset_GaJuSi = TensorDataset(X_val_person_GaJuSi, y_val_person_GaJuSi)
        test_person_dataset_GaJuSi = TensorDataset(X_test_person_GaJuSi, y_test_person_GaJuSi)

        moe_model = HardMoE(experts=expert_model_map.values(), gate=gate_model)

        print_h("EVALUATION ON PERSON DATA BY MAJORITY VOTING", 64)
        (
            _, 
            acc_person_majority_voting, 
            f1_person_majority_voting, 
            precision_person_majority_voting, 
            recall_person_majority_voting, 
            cm_person_majority_voting, 
            _, _, _, 
            fpr_multiclass_list_person_majority_voting, 
            tpr_multiclass_list_person_majority_voting, 
            roc_auc_multiclass_list_person_majority_voting, 
            roc_auc_multiclass_avg_person_majority_voting
        ) = eval_person_majority_voting(
            moe_model, 
            val_person_dataset_GaJuSi, 
            criterion=None, 
            average='weighted',
            window_size=window_size, 
            debug=False
        )
        print("acc:", acc_person_majority_voting)
        print("f1:", f1_person_majority_voting)
        print("precision:", precision_person_majority_voting)
        print("recall:", recall_person_majority_voting)
        print("cm:\n", np.array(cm_person_majority_voting))
        print()

        moe_metrics = update_metrics(moe_metrics, {
            'acc': acc_person_majority_voting,
            'f1': f1_person_majority_voting,
            'precision': precision_person_majority_voting,
            'recall': recall_person_majority_voting,
            'cm': cm_person_majority_voting,
            'fpr_multiclass_list': fpr_multiclass_list_person_majority_voting, 
            'tpr_multiclass_list': tpr_multiclass_list_person_majority_voting, 
            'roc_auc_multiclass_list': roc_auc_multiclass_list_person_majority_voting,
            'roc_auc_multiclass_avg': roc_auc_multiclass_avg_person_majority_voting,
        })

        # DEBUG: Test for only one fold
        # break
    
    # ================================================================================================================================
    # MoE MODEL METRICS
    # ================================================================================================================================
    print_h("MoE METRICS", 128)
    save_metrics_to_json(moe_metrics, general_metrics_dir, '_MoE.json')
    pprint(moe_metrics, sort_dicts=False)

    # ================================================================================================================================
    # MoE MODEL ROC CURVES
    # ================================================================================================================================
    plot_k_fold_roc_curves_multiclass_v2(
        fpr_folds=moe_metrics['fpr_multiclass_list']['folds'],
        tpr_folds=moe_metrics['tpr_multiclass_list']['folds'],
        auc_folds=moe_metrics['roc_auc_multiclass_list']['folds'],
        class_names=["Healthy", "S-2", "S-2.5", "S-3"],
        save_dir=roc_curves_dir,
        show=False,
    )

    # ================================================================================================================================
    # MoE MODEL CONFUSION MATRIX
    # ================================================================================================================================
    plot_k_fold_cm(
        moe_metrics['cm']['folds'], 
        class_names=["Healthy", "S-2", "S-2.5", "S-3"],
        save_dir=cm_dir,
        show=False,
    )

    # ================================================================================================================================
    # GATE MODEL METRICS
    # ================================================================================================================================
    print_h("GATE METRICS", 128)
    save_metrics_to_json(gate_metrics, general_metrics_dir, 'gate.json')
    print()
    pprint(gate_metrics)

    # ================================================================================================================================
    # EXPERT-Ga MODEL METRICS
    # ================================================================================================================================
    print_h("EXPERT-Ga METRICS", 128)
    save_metrics_to_json(expert_metrics['Ga'], general_metrics_dir, 'Ga.json')
    print()
    pprint(expert_metrics['Ga'])

    # ================================================================================================================================
    # EXPERT-Ju MODEL METRICS
    # ================================================================================================================================
    print_h("EXPERT-Ju METRICS", 128)
    save_metrics_to_json(expert_metrics['Ju'], general_metrics_dir, 'Ju.json')
    print()
    pprint(expert_metrics['Ju'])

    # ================================================================================================================================
    # EXPERT-Si MODEL METRICS
    # ================================================================================================================================
    print_h("EXPERT-Si METRICS", 128)
    save_metrics_to_json(expert_metrics['Si'], general_metrics_dir, 'Si.json')
    print()
    pprint(expert_metrics['Si'])

if __name__ == '__main__':
    fire.Fire(main)