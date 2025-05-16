import os
import fire
import numpy as np
import torch
from torch.utils.data import TensorDataset
from pprint import pprint

from src.utils import (
    # Old utils
    print_h, eval_person_majority_voting, 

    # New utils
    set_seed, get_device, init_model, init_metrics, update_metrics, save_metrics_to_json,
    plot_k_fold_roc_curves_multiclass_v2, plot_k_fold_cm,
)

def main(
    model_path,
    fold_i_dir_Ga,
    fold_i_dir_Ju,
    fold_i_dir_Si,
    seed = 69,

    # Evaluation parameters
    # WARNING: Still hard-coded!
    n_feat = 16,
    n_class = 4,
    window_size = 500,
    max_vgrf_data_len = 25_000,
):
    # Set seed and device
    set_seed(seed)
    device = get_device()
    print("Device:", device)

    # Get run name, and model name and bidirectional parameter
    run_name = model_path.split('/')[-2]
    print("Run name:", run_name)
    model_name = model_path.split('/')[-2].split('_non_moe')[0]
    if 'bidirectional' in model_name:
        bidirectional = True
        model_name = model_name.replace('_bidirectional', '')
    else:
        bidirectional = False
    print("Model name:", model_name)
    print("Bidirectional:", bidirectional)

    # Set up single fold data mapping
    fold_i_dir_map = {
        'Ga': fold_i_dir_Ga,
        'Ju': fold_i_dir_Ju,
        'Si': fold_i_dir_Si,
    }

    # Get fold number
    i_folds_data = [int(fold_i_dir.split('fold_')[-1]) for fold_i_dir in fold_i_dir_map.values()]
    i_fold_checkpoint = int(model_path.split('fold_')[-1].replace('.pth', ''))
    i_folds = i_folds_data + [i_fold_checkpoint]
    assert len(set(i_folds)) == 1, f"Fold numbers are inconsistent: {({'data': i_folds_data, 'checkpoint': i_fold_checkpoint})}"
    i_fold = i_folds[0]
    print("Fold number:", i_fold)

    # Set up evaluation directories
    general_metrics_dir = f'evaluations/{run_name}/_general_metrics'
    cm_dir = f'evaluations/{run_name}/cm'
    roc_curves_dir = f'evaluations/{run_name}/roc_curves'

    print("Evaluation general metrics save directory:", general_metrics_dir)
    print("Evaluation confusion matrix save directory:", cm_dir)
    print("Evaluation ROC curves save directory:", roc_curves_dir)

    metrics = init_metrics(['acc', 'f1', 'precision', 'recall', 'cm', 
                        'fpr_multiclass_list', 'tpr_multiclass_list', 'roc_auc_multiclass_list', 'roc_auc_multiclass_avg'])

    print_h(f"FOLD {i_fold}", 128)

    X_val_person_GaJuSi = torch.empty(0, max_vgrf_data_len, n_feat).float()
    y_val_person_GaJuSi = torch.empty(0).long()

    X_test_person_GaJuSi = torch.empty(0, max_vgrf_data_len, n_feat).float()
    y_test_person_GaJuSi = torch.empty(0).long()

    for study, fold_i_dir in fold_i_dir_map.items():
        X_val_person = torch.tensor(np.load(os.path.join(fold_i_dir, f'X_val_person.npy'))).float()
        y_val_person = torch.tensor(np.load(os.path.join(fold_i_dir, f'y_val_person.npy'))).long()
        X_val_person_GaJuSi = torch.cat((X_val_person_GaJuSi, X_val_person), dim=0)
        y_val_person_GaJuSi = torch.cat((y_val_person_GaJuSi, y_val_person), dim=0)

        X_test_person = torch.tensor(np.load(os.path.join(fold_i_dir, f'X_test_person.npy'))).float()
        y_test_person = torch.tensor(np.load(os.path.join(fold_i_dir, f'y_test_person.npy'))).long()
        X_test_person_GaJuSi = torch.cat((X_test_person_GaJuSi, X_test_person), dim=0)
        y_test_person_GaJuSi = torch.cat((y_test_person_GaJuSi, y_test_person), dim=0)

    print_h("NON-MoE MODEL", 96)

    val_person_dataset_GaJuSi = TensorDataset(X_val_person_GaJuSi, y_val_person_GaJuSi)
    test_person_dataset_GaJuSi = TensorDataset(X_test_person_GaJuSi, y_test_person_GaJuSi)

    model = init_model(model_name, device, c_in=n_feat, c_out=n_class, seq_len=window_size, bidirectional=bidirectional)

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
        model, 
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

    metrics = update_metrics(metrics, {
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

    print_h("NON-MoE MODEL METRICS", 128)
    save_metrics_to_json(metrics, general_metrics_dir, f'_non_moe.json')
    pprint(metrics, sort_dicts=False)
    print()

    plot_k_fold_roc_curves_multiclass_v2(
        fpr_folds=metrics['fpr_multiclass_list']['folds'],
        tpr_folds=metrics['tpr_multiclass_list']['folds'],
        auc_folds=metrics['roc_auc_multiclass_list']['folds'],
        class_names=["Healthy", "S-2", "S-2.5", "S-3"],
        save_dir=roc_curves_dir,
        i_folds=[i_fold],
        show=False,
    )
    plot_k_fold_cm(
        metrics['cm']['folds'], 
        class_names=["Healthy", "S-2", "S-2.5", "S-3"],
        save_dir=cm_dir,
        figsize=(25, 5),
        i_folds=[i_fold],
        show=False,
    )
    print()

if __name__ == '__main__':
    fire.Fire(main)