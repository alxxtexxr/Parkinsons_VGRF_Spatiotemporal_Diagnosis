# RNN-InceptionTime-MoE Model for Spatiotemporal Diagnosis of Parkinson’s Disease on VGRF Data

## 1. Data Preprocessing
```bash
python data_preprocessing.py \
--study=<DATASET_STUDY> \
--k_fold=<NUMBER_OF_FOLDS> \
--window_size=<WINDOW_SIZE> \
--stride_size=<STRIDE_SIZE>
```
Parameters:
- `study`: The source study of the dataset. Options: `Ga`, `Ju`, or `Si` .
- `k_fold`: The number of folds used for K-Folds Cross-Validation.
- `window_size`: The size of the sliding window applied to the data.
- `stride_size`: The Step size (stride) between consecutive windows.

## 2. Expert Model Training
```bash
python RNNInceptionTime_expert_training.py \
--k_fold_dir=<PREPROCESSED_DATA_DIRECTORY_PATH> \
--n_epoch=<NUMBER_OF_EPOCHS>
```
Parameters:
- `k_fold_dir`: The preprocessed data directory path for a specific dataset study (`Ga`, `Ju`, or `Si` ). Examples:
  - `Ga_k10_w500_s500_197001010101011`
  - `Ju_k10_w500_s250_197001010101012`
  - `Si_k10_w250_s250_197001010101013`
- `n_epoch`: The number of training epochs.

## 3. Gate Model Training
```bash
python RNNInceptionTimeGate_training.py \
--k_fold_dir_Ga=<PREPROCESSED_Ga_DATA_DIRECTORY_PATH_FOR_Ga> \
--k_fold_dir_Ju=<PREPROCESSED_Ju_DATA_DIRECTORY_PATH_FOR_Ju> \
--k_fold_dir_Si=<PREPROCESSED_Si_DATA_DIRECTORY_PATH_FOR_Si> \
--expert_model_dir_Ga=<EXPERT_MODEL_CHECKPOINT_DIRECTORY_PATH_FOR_Ga> \
--expert_model_dir_Ju=<EXPERT_MODEL_CHECKPOINT_DIRECTORY_PATH_FOR_Ju> \
--expert_model_dir_Si=<EXPERT_MODEL_CHECKPOINT_DIRECTORY_PATH_FOR_Si> \
--n_epoch=<NUMBER_OF_EPOCHS>
```
Parameters:
- `k_fold_dir_Ga`: The preprocessed data directory path for dataset study `Ga`.
- `k_fold_dir_Ju`: The preprocessed data directory path for dataset study `Ju`.
- `k_fold_dir_Si`: The preprocessed data directory path for dataset study `Si`.
- `expert_model_dir_Ga`: The expert model checkpoint directory path for dataset study `Ga`.
- `expert_model_dir_Ju`: The expert model checkpoint directory path for dataset study `Ju`.
- `expert_model_dir_Si`: The expert model checkpoint directory path for dataset study `Si`.
- `n_epoch`: The number of training epochs.

## 4. MoE Model Evaluation
```bash
python RNNInceptionTimeMoE_evaluation.py \
--k_fold_dir_Ga=<PREPROCESSED_Ga_DATA_DIRECTORY_PATH_FOR_Ga> \
--k_fold_dir_Ju=<PREPROCESSED_Ju_DATA_DIRECTORY_PATH_FOR_Ju> \
--k_fold_dir_Si=<PREPROCESSED_Si_DATA_DIRECTORY_PATH_FOR_Si> \
--expert_model_dir_Ga=<EXPERT_MODEL_CHECKPOINT_DIRECTORY_PATH_FOR_Ga> \
--expert_model_dir_Ju=<EXPERT_MODEL_CHECKPOINT_DIRECTORY_PATH_FOR_Ju> \
--expert_model_dir_Si=<EXPERT_MODEL_CHECKPOINT_DIRECTORY_PATH_FOR_Si> \
--gate_model_dir=<GATE_MODEL_CHECKPOINT_DIRECTORY_PATH>
```
Parameters:
- `k_fold_dir_Ga`: The preprocessed data directory path for dataset study `Ga`.
- `k_fold_dir_Ju`: The preprocessed data directory path for dataset study `Ju`.
- `k_fold_dir_Si`: The preprocessed data directory path for dataset study `Si`.
- `expert_model_dir_Ga`: The expert model checkpoint directory path for dataset study `Ga`.
- `expert_model_dir_Ju`: The expert model checkpoint directory path for dataset study `Ju`.
- `expert_model_dir_Si`: The expert model checkpoint directory path for dataset study `Si`.
- `gate_model_dir`: The gate model checkpoint directory path.
