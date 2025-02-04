import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional

import sys
sys.path.append('../')

from eval import evaluate, bootstrap
from zero_shot import make, make_true_labels, run_softmax_eval

# ----- DIRECTORIES ------ #
cxr_filepath: str = '/home/vault/iwi5/iwi5207h/new/CheXzero/dataset/new_testing_without_lateral.h5' # filepath of chest x-ray images (.h5)
cxr_true_labels_path: Optional[str] = '/home/hpc/iwi5/iwi5207h/Thesis/dataset_csv/testing_data_without_lateral_only_nec.csv' # (optional for evaluation) if labels are provided, provide path
model_dir: str = '../checkpoints/chexzero_weights' # where pretrained models are saved (.pt) 
predictions_dir: Path = Path('../predictions') # where to save predictions
cache_dir: str = predictions_dir / "cached" # where to cache ensembled predictions

context_length: int = 77

# ------- LABELS ------  #
'''cxr_labels: List[str] =  [
        'Adenopathy', 'Atelectasis', 'Azygos Lobe', 'Calcification of the Aorta', 'Cardiomegaly',
        'Clavicle Fracture', 'Consolidation', 'Edema', 'Emphysema', 'Enlarged Cardiomediastinum',
        'Fibrosis', 'Fissure', 'Fracture', 'Granuloma', 'Hernia', 'Hydropneumothorax', 'Infarction',
        'Infiltration', 'Kyphosis', 'Lobar Atelectasis', 'Lung Lesion', 'Lung Opacity', 'Mass', 'Nodule',
        'Normal', 'Pleural Effusion', 'Pleural Other', 'Pleural Thickening', 'Pneumomediastinum', 'Pneumonia',
        'Pneumoperitoneum', 'Pneumothorax', 'Pulmonary Embolism', 'Pulmonary Hypertension', 'Rib Fracture',
        'Round(ed) Atelectasis', 'Subcutaneous Emphysema', 'Support Devices', 'Tortuous Aorta', 'Tuberculosis'
    ]'''
'''cxr_labels: List[str] = ["Aortic enlargement", "Atelectasis", "Cardiomegaly", "Calcification", 
    "Clavicle fracture", "Consolidation", "Edema", "Emphysema", "Enlarged PA", 
    "ILD", "Infiltration", "Lung cavity", 
    "Lung cyst", "Lung Opacity", "Mediastinal shift", "Nodule/Mass", 
    "Pulmonary fibrosis", "Pneumothorax", "Pleural thickening", 
    "Pleural effusion", "Rib fracture", "Other lesion", "Lung tumor", 
    "Pneumonia", "Tuberculosis", "Other disease", 
    "COPD", "No finding"]'''
cxr_labels: List[str] = ['Lobar Atelectasis','Round(ed) Atelectasis','Clavicle Fracture','Azygos Lobe','Pleural Other','Pneumoperitoneum',
                  'Kyphosis','Hydropneumothorax','Infarction','Pneumomediastinum','Pulmonary Hypertension','Fibrosis']

# ---- TEMPLATES ----- # 
cxr_pair_template: Tuple[str] = ("{}", "no {}")

# ----- MODEL PATHS ------ #
# If using ensemble, collect all model paths
model_paths = []
for subdir, dirs, files in os.walk(model_dir):
    for file in files:
        full_dir = os.path.join(subdir, file)
        model_paths.append(full_dir)
        
print(model_paths)


# Run the model on the data set using ensembled models
def ensemble_models(
    model_paths: List[str], 
    cxr_filepath: str, 
    cxr_labels: List[str], 
    cxr_pair_template: Tuple[str], 
    cache_dir: str = None, 
    save_name: str = None,
) -> Tuple[List[np.ndarray], np.ndarray]: 
    """
    Given a list of `model_paths`, ensemble model and return
    predictions. Caches predictions at `cache_dir` if location provided.

    Returns a list of each model's predictions and the averaged
    set of predictions.
    """

    predictions = []
    model_paths = sorted(model_paths)  # ensure consistency 
    for path in model_paths:  # for each model
        model_name = Path(path).stem

        # load in model and `torch.DataLoader`
        model, loader = make(
            model_path=path, 
            cxr_filepath=cxr_filepath, 
        ) 
        
        # path to the cached prediction
        if cache_dir is not None:
            if save_name is not None: 
                cache_path = Path(cache_dir) / f"{save_name}_{model_name}.npy"
            else: 
                cache_path = Path(cache_dir) / f"{model_name}.npy"

        # if prediction already cached, don't recompute prediction
        if cache_dir is not None and os.path.exists(cache_path): 
            print(f"Loading cached prediction for {model_name}")
            y_pred = np.load(cache_path)
        else:  # cached prediction not found, compute preds
            print(f"Inferring model {path}")
            y_pred = run_softmax_eval(model, loader, cxr_labels, cxr_pair_template)
            if cache_dir is not None: 
                Path(cache_dir).mkdir(exist_ok=True, parents=True)
                np.save(file=cache_path, arr=y_pred)
        predictions.append(y_pred)
    
    # compute average predictions
    y_pred_avg = np.mean(predictions, axis=0)
    
    return predictions, y_pred_avg


# Ensemble models and get predictions
predictions, y_pred_avg = ensemble_models(
    model_paths=model_paths, 
    cxr_filepath=cxr_filepath, 
    cxr_labels=cxr_labels, 
    cxr_pair_template=cxr_pair_template, 
    cache_dir=cache_dir,
)

# Save averaged preds
#pred_name = "chexpert_preds_original.npy"  # add name of preds
predictions_dir = '/home/woody/iwi5/iwi5207h/metric_learning/CheXzero/predictions/chexzero_check_org_rare.npy'
np.save(file=predictions_dir, arr=y_pred_avg)

# Make test_true
test_pred = y_pred_avg
test_true = make_true_labels(cxr_true_labels_path=cxr_true_labels_path, cxr_labels=cxr_labels)

# Evaluate model
cxr_results = evaluate(test_pred, test_true, cxr_labels)
cxr_results=cxr_results.mean()


# Bootstrap evaluations for 95% confidence intervals
bootstrap_results = bootstrap(test_pred, test_true, cxr_labels)

# Save cxr_results and bootstrap_results to CSV
'''cxr_results_df = pd.DataFrame(cxr_results)
bootstrap_results_df = pd.DataFrame(bootstrap_results)

# Define paths for saving results
cxr_results_path = '/home/vault/iwi5/iwi5207h/not_frozen/CheXzero/predictions/ cxr_results.csv'
bootstrap_results_path =' /home/vault/iwi5/iwi5207h/not_frozen/CheXzero/predictions / bootstrap_results.csv'

# Save to CSV
cxr_results_df.to_csv(cxr_results_path, index=False)
bootstrap_results_df.to_csv(bootstrap_results_path, index=False)

print(f"CXR results saved to {cxr_results_path}")
print(f"Bootstrap results saved to {bootstrap_results_path}")'''

print(cxr_results)
print(bootstrap_results)
