#!/bin/python
import os

import pandas as pd
import numpy as np

import shap

from xgboost import XGBClassifier
from config import Config


def shap_explain(clf, X, outcome):
    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(X)
    shap.summary_plot(shap_vals, X)

    proceed = input('Save? ')
    if proceed != '':
        os.makedirs('SHAP', exist_ok=True)
        shap_interactions = explainer.shap_interaction_values(X)
        pd.to_pickle(X, f'SHAP/{outcome}_patients.pickle', protocol=4)
        pd.to_pickle(shap_vals, f'SHAP/{outcome}_shap_vals.pickle', protocol=4)
        pd.to_pickle(shap_interactions, f'SHAP/{outcome}_shap_interactions.pickle', protocol=4)


def shap_dump(clf, X, clf_desc, dataset_desc, fold, suffix):
    outdir = Config.shap_dir

    os.makedirs(outdir, exist_ok=True)
    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(X)
    shap_interactions = explainer.shap_interaction_values(X)

    X.to_pickle(
        f'{outdir}/{dataset_desc}_{suffix}_{fold}.pickle',
        protocol=4)
    np.save(
        f'{outdir}/{clf_desc}_{dataset_desc}_{fold}_shap_values.npy',
        shap_vals)
    np.save(
        f'{outdir}/{clf_desc}_{dataset_desc}_{fold}_shap_interactions.npy',
        shap_interactions)
