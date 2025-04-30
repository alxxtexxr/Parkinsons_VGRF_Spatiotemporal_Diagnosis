import os
import itertools
import random
import math
import fire
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import StratifiedShuffleSplit

from src.utils import set_seed, print_h, check_dataset_info, get_vgrf_window_data, resample_linear_interpolation
from src.data import Dataset_v2

DATA_DIR = 'data'
VGRF_DATA_DIR = DATA_DIR + '/gait-in-parkinsons-disease-1.0.0'

def main(
    study,   # 'Ga' | 'Ju' | 'Si'
    k_fold,
    window_size,
    stride_size,
    
    wo_synth = False,   # For ablation study, set to True to preprocess w/o synthetic data
    w_anomaly = False,  # For ablation study, set to True to preprocess w/ anomaly data
    anomaly_ids = None,
    seed = 69,

    # Data parameters
    vgrf_data_dir = VGRF_DATA_DIR,
    demographics_data_path = VGRF_DATA_DIR + '/demographics.xls',
    preprocessed_data_dir = DATA_DIR + '/preprocessed',

    # Training parameters
    n_feat = 16,
    max_vgrf_data_len = 25_000,
    test_person_split = 0.15,
    test_window_split = 0.15,
    val_person = True,
):
    set_seed(seed)
    print()

    demographics_df = pd.read_excel(demographics_data_path)
    demographics_df['HoehnYahr'] = demographics_df['HoehnYahr'].fillna(0)   # Fill N/A values in 'HoehnYahr' column
    demographics_df = demographics_df[demographics_df['Study'] == study]

    # Remove anomaly data
    if not w_anomaly:
        if anomaly_ids is None:
            anomaly_ids = []

        demographics_df = demographics_df[~demographics_df['ID'].isin(anomaly_ids)]
        print(f"Anomaly data with IDS {anomaly_ids} have been removed.")
        print()
    else:
        print("Warning: anomaly data will not be removed.")
        print()

    min_vgrf_data_len = (5-1)*stride_size + window_size
    dataset_person = Dataset_v2(demographics_df, vgrf_data_dir, n_feat, 
                                min_vgrf_data_len, max_vgrf_data_len, window_size, stride_size, 
                                type='person')
    X = dataset_person.X
    y = dataset_person.y

    print_h("PERSON DATASET INFO", 32)
    check_dataset_info(X, y)
    print()

    if wo_synth:
        print("Warning: synthetic data will not be generated.")
        print()

    data_folds = []

    idxs0_test = []
    idxs1_test = []
    idxs2_test = []
    idxs3_test = []

    idxs0_val = []
    idxs1_val = []
    idxs2_val = []
    idxs3_val = []

    for i_fold in range(k_fold):
        print_h(f"FOLD {i_fold+1}")

        X0 = X[y == 0]
        y0 = y[y == 0]
        X1 = X[y == 1]
        y1 = y[y == 1]
        X2 = X[y == 2]
        y2 = y[y == 2]
        X3 = X[y == 3]
        y3 = y[y == 3]
        
        print("Total data:", X.shape[0])
        print()

        z = 1 + int(val_person)

        print("Total data with class 0:", X0.shape[0])
        n0_test = math.floor(X0.shape[0] * test_person_split)
        # n0_test = math.floor((X0.shape[0] / k_fold) / z) if n0_test * z * k_fold > X0.shape[0] else n0_test
        # idx0_test = random.sample(list(set(range(X0.shape[0])) - set(idxs0_test)), n0_test)
        idx0_test = random.sample(list(set(range(X0.shape[0]))), n0_test)
        idxs0_test += idx0_test
        X0_test = X0[idx0_test]
        y0_test = y0[idx0_test]
        print("Indices of data with class 0 for testing:", idx0_test, "— Total data:", n0_test)

        if val_person:
            # idx0_val = random.sample(list(set(range(X0.shape[0])) - set(idxs0_val) - set(idxs0_test)), n0_test)
            idx0_val = random.sample(list(set(range(X0.shape[0])) - set(idx0_test)), n0_test)
            idxs0_val += idx0_val
            X0_val = X0[idx0_val]
            y0_val = y0[idx0_val]
            print("Indices of data with class 0 for validation:", idx0_val, "— Total data:", n0_test)

        mask = torch.ones(X0.shape[0], dtype=torch.bool)
        mask[idx0_test] = False
        if val_person:
            mask[idx0_val] = False
        X0_train = X0[mask]
        y0_train = y0[mask]
        print("Total data with class 0 for training:", X0_train.shape[0])
        print()

        print("Total data with class 1:", X1.shape[0])
        n1_test = math.floor(X1.shape[0] * test_person_split)
        # n1_test = math.floor((X1.shape[0] / k_fold) / z) if n1_test * z * k_fold > X1.shape[0] else n1_test
        # idx1_test = random.sample(list(set(range(X1.shape[0])) - set(idxs1_test)), n1_test)
        idx1_test = random.sample(list(set(range(X1.shape[0]))), n1_test)
        idxs1_test += idx1_test
        X1_test = X1[idx1_test]
        y1_test = y1[idx1_test]
        print("Indices of data with class 1 for testing:", idx1_test, "— Total data:", n1_test)

        if val_person:
            # idx1_val = random.sample(list(set(range(X1.shape[0])) - set(idxs1_val) - set(idxs1_test)), n1_test)
            idx1_val = random.sample(list(set(range(X1.shape[0])) - set(idx1_test)), n1_test)
            idxs1_val += idx1_val
            X1_val = X1[idx1_val]
            y1_val = y1[idx1_val]
            print("Indices of data with class 1 for validation:", idx1_val, "— Total data:", n1_test)

        mask = torch.ones(X1.shape[0], dtype=torch.bool)
        mask[idx1_test] = False
        if val_person:
            mask[idx1_val] = False
        X1_train = X1[mask]
        y1_train = y1[mask]
        print("Total data with class 1 for training:", X1_train.shape[0])
        print()

        print("Total data with class 2:", X2.shape[0])
        n2_test = math.floor(X2.shape[0] * test_person_split)
        # n2_test = math.floor((X2.shape[0] / k_fold) / z) if n2_test * z * k_fold > X2.shape[0] else n2_test
        # idx2_test = random.sample(list(set(range(X2.shape[0])) - set(idxs2_test)), n2_test)
        idx2_test = random.sample(list(set(range(X2.shape[0]))), n2_test)
        idxs2_test += idx2_test
        X2_test = X2[idx2_test]
        y2_test = y2[idx2_test]
        print("Indices of data with class 2 for testing:", idx2_test, "— Total data:", n2_test)

        if val_person:
            # idx2_val = random.sample(list(set(range(X2.shape[0])) - set(idxs2_val) - set(idxs2_test)), n2_test)
            idx2_val = random.sample(list(set(range(X2.shape[0])) - set(idx2_test)), n2_test)
            idxs2_val += idx2_val
            X2_val = X2[idx2_val]
            y2_val = y2[idx2_val]
            print("Indices of data with class 2 for validation:", idx2_val, "— Total data:", n2_test)

        mask = torch.ones(X2.shape[0], dtype=torch.bool)
        mask[idx2_test] = False
        if val_person:
            mask[idx2_val] = False
        X2_train = X2[mask]
        y2_train = y2[mask]
        print("Total data with class 2 for training:", X2_train.shape[0])
        print()

        print("Total data with class 3:", X3.shape[0])
        n3_test = math.floor(X3.shape[0] * test_person_split)
        # n3_test = math.floor((X3.shape[0] / k_fold) / z) if n3_test * z * k_fold > X3.shape[0] else n3_test
        # idx3_test = random.sample(list(set(range(X3.shape[0])) - set(idxs3_test)), n3_test)
        idx3_test = random.sample(list(set(range(X3.shape[0]))), n3_test)
        idxs3_test += idx3_test
        X3_test = X3[idx3_test]
        y3_test = y3[idx3_test]
        print("Indices of data with class 3 for testing:", idx3_test, "— Total data:", n3_test)

        if val_person:
            # idx3_val = random.sample(list(set(range(X3.shape[0])) - set(idxs3_val) - set(idxs3_test)), n3_test)
            idx3_val = random.sample(list(set(range(X3.shape[0])) - set(idx3_test)), n3_test)
            idxs3_val += idx3_val
            X3_val = X3[idx3_val]
            y3_val = y3[idx3_val]
            print("Indices of data with class 3 for validation:", idx3_val, "— Total data:", n3_test)

        mask = torch.ones(X3.shape[0], dtype=torch.bool)
        mask[idx3_test] = False
        if val_person:
            mask[idx3_val] = False
        X3_train = X3[mask]
        y3_train = y3[mask]
        print("Total data with class 3 for training:", X3_train.shape[0])
        print()

        X_test = torch.cat([X0_test, X1_test, X2_test, X3_test], dim=0)
        y_test = torch.cat([y0_test, y1_test, y2_test, y3_test], dim=0)
        assert X_test.shape[0] == y_test.shape[0], f"Mismatch between number of samples in X_test ({X_test.shape[0]}) and y_test ({y_test.shape[0]})"
        
        # Randomize rows
        random_idxs = torch.randperm(y_test.shape[0])
        X_test = X_test[random_idxs]
        y_test = y_test[random_idxs]
        print("Total data for testing:", X_test.shape[0])

        if val_person:
            X_val = torch.cat([X0_val, X1_val, X2_val, X3_val], dim=0)
            y_val = torch.cat([y0_val, y1_val, y2_val, y3_val], dim=0)

            # Randomize rows
            assert X_val.shape[0] == y_val.shape[0]
            random_idxs = torch.randperm(y_val.shape[0])
            X_val = X_val[random_idxs]
            y_val = y_val[random_idxs]
            print("Total data for validation:", X_val.shape[0])
        else:
            X_val = None
            y_val = None

        X_train = torch.cat([X0_train, X1_train, X2_train, X3_train], dim=0)
        y_train = torch.cat([y0_train, y1_train, y2_train, y3_train], dim=0)
        assert X_train.shape[0] == y_train.shape[0], f"Mismatch between number of samples in X_train ({X_train.shape[0]}) and y_train ({y_train.shape[0]})"
        
        # Randomize rows
        random_idxs = torch.randperm(y_train.shape[0])
        X_train = X_train[random_idxs]
        y_train = y_train[random_idxs]
        print("Total data for training:", X_train.shape[0])

        # ================================================================================================
        # DATA RESAMPLING
        # ================================================================================================
        if not wo_synth:
            X0_train = X_train[y_train == 0]
            y0_train = y_train[y_train == 0]

            X1_train = X_train[y_train == 1]
            y1_train = y_train[y_train == 1]

            X2_train = X_train[y_train == 2]
            y2_train = y_train[y_train == 2]

            X0_train_new_size = 30
            X0_train_new = resample_linear_interpolation(X0, n=X0_train_new_size)
            y0_train_new = torch.tensor([0] * X0_train_new_size, dtype=torch.long)

            X1_train_new_size = 30
            X1_train_new = resample_linear_interpolation(X1, n=X1_train_new_size)
            y1_train_new = torch.tensor([1] * X1_train_new_size, dtype=torch.long)

            X2_train_new_size = 25
            X2_train_new = resample_linear_interpolation(X2, n=X2_train_new_size)
            y2_train_new = torch.tensor([2] * X2_train_new_size, dtype=torch.long)

            X_train = torch.cat([X_train, X0_train_new, X1_train_new, X2_train_new], dim=0)
            y_train = torch.cat([y_train, y0_train_new, y1_train_new, y2_train_new], dim=0)
            assert X_train.shape[0] == y_train.shape[0], f"Mismatch between number of samples in X_train ({X_train.shape[0]}) and y_train ({y_train.shape[0]})"

            print("Total data for training w/ synthetic data:", X_train.shape[0])
        # ================================================================================================

        data_folds.append(dict(
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            X_val=X_val,
            y_val=y_val,
        ))

        print()
    
    save_dir = preprocessed_data_dir + f'/{study}_k{k_fold}_w{window_size}_s{stride_size}_f{n_feat}{'_w_anomaly' if w_anomaly else ''}{'_wo_synth' if wo_synth else ''}_v{datetime.now().strftime("%Y%m%d%H%M%S")}'
    print("Save directory:", save_dir)
    print()

    for i_fold, data_fold in enumerate(data_folds):
        print_h(f"FOLD {i_fold+1}")
        
        X_train, y_train, X_test, y_test, X_val, y_val = data_fold.values()

        X_test_person, y_test_person = X_test, y_test
        X_val_person, y_val_person = X_val, y_val
        X_window, y_window, _ = get_vgrf_window_data(X_train, y_train, window_size, stride_size, zeros_filter_thres=1.0)

        print_h("TESTING PERSON DATASET INFO", 64)
        check_dataset_info(X_test_person, y_test_person)
        print()

        if val_person:
            print_h("VALIDATION PERSON DATASET INFO", 64)
            check_dataset_info(X_val_person, y_val_person)
            print()

        # ================================================================================================
        # DATA STRATIFICATION
        # ================================================================================================
        print_h(f"DATA STRATISFICATION", 96)

        sss1 = StratifiedShuffleSplit(n_splits=k_fold, test_size=(test_window_split * 2), random_state=seed)
        sss1_split = sss1.split(X=X_window, y=y_window)

        train_idxs, val_test_idxs = next(itertools.islice(sss1_split, i_fold, None)) # Get i-th fold from the stratification
        X_train_window = X_window[train_idxs]
        y_train_window = y_window[train_idxs]
        X_val_test_window = X_window[val_test_idxs]
        y_val_test_window = y_window[val_test_idxs]

        sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        sss2_split = sss2.split(X=X_val_test_window, y=y_val_test_window)

        val_idxs, test_idxs = next(iter(sss2_split))
        X_val_window = X_val_test_window[val_idxs]
        y_val_window = y_val_test_window[val_idxs]
        X_test_window = X_val_test_window[test_idxs]
        y_test_window = y_val_test_window[test_idxs]

        print_h("TRAINING WINDOW DATASET INFO", 64)
        check_dataset_info(X_train_window, y_train_window)
        print()

        print_h("VALIDATION WINDOW DATASET INFO", 64)
        check_dataset_info(X_val_window, y_val_window)
        print()

        print_h("TESTING WINDOW DATASET INFO", 64)
        check_dataset_info(X_test_window, y_test_window)
        print()

        # ================================================================================================
        # DATA SAVING
        # ================================================================================================
        # print_h("DATA SAVING", 96)
        
        fold_i_dir = os.path.join(save_dir, f'fold_{i_fold+1}')
        os.makedirs(fold_i_dir, exist_ok=True)
        
        # Save person datasets
        # np.save(os.path.join(fold_i_dir, 'X_train_person.npy'), X_train_person.detach().cpu().numpy())
        # np.save(os.path.join(fold_i_dir, 'y_train_person.npy'), y_train_person.detach().cpu().numpy())
        
        if val_person:
            np.save(os.path.join(fold_i_dir, 'X_val_person.npy'), X_val_person.detach().cpu().numpy())
            np.save(os.path.join(fold_i_dir, 'y_val_person.npy'), y_val_person.detach().cpu().numpy())
        np.save(os.path.join(fold_i_dir, 'X_test_person.npy'), X_test_person.detach().cpu().numpy())
        np.save(os.path.join(fold_i_dir, 'y_test_person.npy'), y_test_person.detach().cpu().numpy())

        # Save window datasets
        np.save(os.path.join(fold_i_dir, 'X_train_window.npy'), X_train_window.detach().cpu().numpy())
        np.save(os.path.join(fold_i_dir, 'y_train_window.npy'), y_train_window.detach().cpu().numpy())
        np.save(os.path.join(fold_i_dir, 'X_val_window.npy'), X_val_window.detach().cpu().numpy())
        np.save(os.path.join(fold_i_dir, 'y_val_window.npy'), y_val_window.detach().cpu().numpy())
        np.save(os.path.join(fold_i_dir, 'X_test_window.npy'), X_test_window.detach().cpu().numpy())
        np.save(os.path.join(fold_i_dir, 'y_test_window.npy'), y_test_window.detach().cpu().numpy())

        # print("Saved in", fold_i_dir)
        # print()

if __name__ == '__main__':
    fire.Fire(main)