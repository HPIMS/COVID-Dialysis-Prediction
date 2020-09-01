#!/bin/python

import os

import pandas as pd
import numpy as np

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score, confusion_matrix, roc_curve
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score, auc, precision_recall_curve

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.impute import KNNImputer

from config import Config
from plotter import shap_explain, shap_dump


class HammerTime:
    def __init__(self, df_train, outcome, day, explain=False):
        # This is ALL that's needed
        self.df = df_train
        self.outcome = outcome

        self.explain = explain
        self.explain_oc_string = f'{outcome}_{day}'

        # Classifier descriptions
        self.clf_descriptions = Config.clf_descriptions

        # Keep these columns
        self.keep_cols = self.df.columns[
            (self.df.count() / self.df.shape[0]) > Config.values_present]
        self.df = self.df[self.keep_cols]

        df_count_c = (self.df.count(axis=1) / self.df.shape[1]) > (Config.values_present - 0.1)
        df_count_c = df_count_c[df_count_c]  # Drops False
        self.df = self.df.reindex(df_count_c.index)

        print('Features:', self.df.shape[1])

        # FINAL classifier - XGB_NotImputed
        self.clf_final = None

        # Get results from here
        self.dict_metrics = {}
        primary_keys = ('cross_val', 'test')
        secondary_keys = self.clf_descriptions.keys()
        tertiary_keys = (
            'acc', 'auroc', 'auprc', 'f1s',
            'sens', 'spec', 'c_matrix', 'roc_curves',  # NOTE Keep the c_matrix for later
            'pr_curves')

        for key in primary_keys:
            self.dict_metrics[key] = {}
            for second_key in secondary_keys:
                if key == 'cross_val' or key != 'cross_val':
                    self.dict_metrics[key][second_key] = {}
                    for third_key in tertiary_keys:
                        self.dict_metrics[key][second_key][third_key] = []

        # Store thresholds here - for calculation of median
        self.dict_thresholds = {i: [] for i in self.clf_descriptions}

        # Start cross validation
        self.cant_touch_this(Config.random_state)

        # Train final classifier
        self.baby_got_back()

    def pick_threshold(self, y_true, pred_proba, clf_desc):
        # Series for thresholds and f1 scores
        s_thresholds = pd.Series([i/100 for i in range(0, 105, 5)])
        # Optimize by f1 score
        s_f1 = pd.Series([f1_score(y_true, pred_proba >= t) for t in s_thresholds])

        # Create a dataframe and check it for the best threshold
        df_t = pd.DataFrame([s_thresholds, s_f1]).T
        df_t.columns = ['THRESHOLD', 'F1S']
        best_threshold = df_t['THRESHOLD'].loc[df_t['F1S'].nlargest(1).index].iloc[0]

        self.dict_thresholds[clf_desc].append(best_threshold)

        return best_threshold

    def get_classifier(self, clf_desc, ratio):
        try:
            clf_hyperparams = self.clf_descriptions[clf_desc]
            if clf_hyperparams is None:
                raise KeyError
        except (KeyError, TypeError):
            # Return default hyperparams
            default_classifiers = {
                'XGB_Imputed': XGBClassifier(
                    n_jobs=-1, gpu_id=0, n_estimators=80,
                    max_delta_step=1),
                'XGB_NotImputed': XGBClassifier(
                    n_jobs=-1, gpu_id=0, n_estimators=80,
                    max_delta_step=1),
                'LASSO': LogisticRegression(penalty='l1', C=0.1, solver='liblinear'),
                'LogisticRegression': LogisticRegression(),
                'RandomForest': RandomForestClassifier()
            }

            return default_classifiers[clf_desc]

        # Now do it by classifier
        if clf_desc in ('XGB_Imputed', 'XGB_NotImputed'):
            df_hyperparams = pd.read_pickle(
                os.path.join(Config.hyperparams_dir, clf_hyperparams))
            dict_hyperparams = df_hyperparams.query(
                'OUTCOME == @self.explain_oc_string').HYPERPARAMETERS.iloc[0]

            clf = XGBClassifier(
                n_jobs=-1,
                gpu_id=0,
                max_delta_step=1,
                n_estimators=dict_hyperparams['n_estimators'],
                learning_rate=dict_hyperparams['learning_rate'],
                max_depth=dict_hyperparams['max_depth'],
                min_child_weight=dict_hyperparams['min_child_weight'],
                gamma=dict_hyperparams['gamma'],
                colsample_bytree=dict_hyperparams['colsample_bytree'])

        elif clf_desc in ('LASSO', 'LogisticRegression'):
            df_hyperparams = pd.read_pickle(
                os.path.join(Config.hyperparams_dir, clf_hyperparams))
            dict_hyperparams = df_hyperparams.query(
                'OUTCOME == @self.explain_oc_string').HYPERPARAMETERS.iloc[0]

            clf = LogisticRegression(n_jobs=-1, C=dict_hyperparams['C'])
            if clf_desc.startswith('LA'):
                clf = LogisticRegression(
                    penalty='l1',
                    solver='liblinear',
                    C=dict_hyperparams['C'])

        return clf

    def imputer(self, df, clf_desc):
        if clf_desc != 'XGB_NotImputed':
            imputed = KNNImputer(
                n_neighbors=5, weights='uniform').fit_transform(df)
            df_imputed = pd.DataFrame(imputed, columns=df.columns, index=df.index)
            return df_imputed

        return df

    def calc_sens_spec(self, cm):
        sens = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        spec = cm[1, 1] / (cm[1, 1] + cm[1, 0])
        return sens, spec

    def cant_touch_this(self, random_state):
        # Training / testing data
        df_X = self.df.drop(self.outcome, axis=1)
        y = self.df[[self.outcome]]

        # Init cross fold validator
        # NOTE - Cross fold validation calculates self performance
        # and median threshold.
        # The final model must have ALL data
        skf = StratifiedKFold(
            Config.validation_folds,
            shuffle=True,
            random_state=random_state)
        zero_array = np.zeros(y.shape[0])

        fold = 0

        for (train, test) in skf.split(zero_array, y):
            # Classification fun for the whole family
            for clf_desc in self.clf_descriptions:

                # Impute - decision to impute occurs within the method
                X = self.imputer(df_X, clf_desc)

                X_train = X.iloc[train]
                y_train = y.iloc[train].values.ravel()

                X_test = X.iloc[test]
                y_test = y.iloc[test].values.ravel()

                # For scale_pos_weight in XGB
                ratio = y[y[self.outcome] == 0].shape[0] // y[y[self.outcome] == 1].shape[0]

                # Fully formed classifier should come from here
                # That includes hyperparameters
                clf = self.get_classifier(clf_desc, ratio)

                clf.fit(X_train, y_train)
                pred_pr = clf.predict_proba(X_test)[:, 1]

                threshold = self.pick_threshold(
                    y_test, pred_pr, clf_desc)

                acc = accuracy_score(y_test, pred_pr >= threshold)
                self.dict_metrics['cross_val'][clf_desc]['acc'].append(acc)

                fpr, tpr, _ = roc_curve(y_test, pred_pr)
                auroc = roc_auc_score(y_test, pred_pr)
                self.dict_metrics['cross_val'][clf_desc]['auroc'].append(auroc)
                self.dict_metrics['cross_val'][clf_desc]['roc_curves'].append((fpr, tpr))

                precision, recall, _ = precision_recall_curve(y_test, pred_pr)
                auprc = auc(recall, precision)
                self.dict_metrics['cross_val'][clf_desc]['auprc'].append(auprc)
                self.dict_metrics['cross_val'][clf_desc]['pr_curves'].append((precision, recall))

                f1_s = f1_score(y_test, pred_pr >= threshold)
                self.dict_metrics['cross_val'][clf_desc]['f1s'].append(f1_s)

                c_matrix = confusion_matrix(y_test, pred_pr >= threshold, labels=[1, 0])
                sens, spec = self.calc_sens_spec(c_matrix)
                self.dict_metrics['cross_val'][clf_desc]['sens'].append(sens)
                self.dict_metrics['cross_val'][clf_desc]['spec'].append(spec)

                if clf_desc in ('XGB_NotImputed',):
                    is_imputed = clf_desc.split('_')[1]
                    # shap_explain(clf, X_test, self.explain_oc_string)
                    shap_dump(
                        clf, X_test, clf_desc,
                        self.explain_oc_string,
                        fold, f'X_test_{is_imputed}')

            fold += 1

    def baby_got_back(self):
        # Train final classifier on the basis of all MSH data
        X = self.df.drop(self.outcome, axis=1)
        X_imputed = self.imputer(X, None)
        y = self.df[[self.outcome]]

        df_calibrate = pd.read_pickle('BestBrier.pickle')
        df_calibrate = df_calibrate[
            df_calibrate['EVENT'].str.contains(self.outcome) &
            df_calibrate['EVENT'].str.contains(self.explain_oc_string[-1:])]  # Should work for 10
        df_calibrate = df_calibrate.reset_index(drop=True)

        self.LASSO_final = self.get_classifier('LASSO', None)
        self.LR_final = self.get_classifier('LogisticRegression', None)
        self.XGBI_final = self.get_classifier('XGB_Imputed', None)
        self.clf_final = self.get_classifier('XGB_NotImputed', None)  # HAS to be XGB_NotImputed
        self.RF_final = self.get_classifier('RandomForest', None)

        # Calibration
        dict_model_order = {
            0: self.LASSO_final,
            1: self.LR_final,
            2: self.XGBI_final,
            3: self.clf_final,
            4: self.RF_final}

        for order, classifier in dict_model_order.items():
            tr_X = X_imputed.copy()
            if order == 3:
                tr_X = X.copy()

            if Config.calibrate_in_training:
                try:
                    if 'Sigmoid' in df_calibrate['CLF'].iloc[order]:
                        dict_model_order[order] = CalibratedClassifierCV(
                            classifier, cv=10, method='sigmoid')
                    elif 'Isotonic' in df_calibrate['CLF'].iloc[order]:
                        dict_model_order[order] = CalibratedClassifierCV(
                            classifier, cv=10, method='isotonic')
                except IndexError:
                    pass

            dict_model_order[order].fit(tr_X, y.values.ravel())

        # Just to make sure
        self.LASSO_final = dict_model_order[0]
        self.LR_final = dict_model_order[1]
        self.XGBI_final = dict_model_order[2]
        self.clf_final = dict_model_order[3]
        self.RF_final = dict_model_order[4]

        # LASSO coefficients for feature importance
        # Only work without calibration in place
        if not Config.calibrate_in_training:
            os.makedirs('LASSOCoef', exist_ok=True)
            df_coef = pd.DataFrame(self.LASSO_final.coef_, columns=X.columns)
            df_coef.to_pickle(
                f'LASSOCoef/LASSOCoefficients{self.explain_oc_string}.pickle', protocol=4)

    def test_validate(self, df, key):
        # Reduce to columns the classifier was trained on
        df = df[self.keep_cols]

        dict_clf = {
            'XGB_NotImputed': self.clf_final,
            'XGB_Imputed': self.XGBI_final,
            'LASSO': self.LASSO_final,
            'LogisticRegression': self.LR_final,
            'RandomForest': self.RF_final
        }

        for clf_desc, clf in dict_clf.items():
            X = df.drop(self.outcome, axis=1)
            X = self.imputer(X, clf_desc)  # Condition present @ the level of the imputer

            y = df[self.outcome].values.ravel()

            pred_pr = clf.predict_proba(X)[:, 1]

            # CRITICAL
            # Get threshold from earlier
            threshold = np.median(self.dict_thresholds[clf_desc])
            threshold = round(threshold, 1)

            acc = accuracy_score(y, pred_pr >= threshold)
            self.dict_metrics[key][clf_desc]['acc'].append(acc)

            fpr, tpr, _ = roc_curve(y, pred_pr)
            auroc = roc_auc_score(y, pred_pr)
            self.dict_metrics[key][clf_desc]['auroc'].append(auroc)
            self.dict_metrics[key][clf_desc]['roc_curves'].append((fpr, tpr))

            precision, recall, _ = precision_recall_curve(y, pred_pr)
            auprc = auc(recall, precision)
            self.dict_metrics[key][clf_desc]['auprc'].append(auprc)
            self.dict_metrics[key][clf_desc]['pr_curves'].append((precision, recall))

            f1_s = f1_score(y, pred_pr >= threshold)
            self.dict_metrics[key][clf_desc]['f1s'].append(f1_s)

            c_matrix = confusion_matrix(y, pred_pr >= threshold, labels=[1, 0])
            self.dict_metrics[key][clf_desc]['c_matrix'].append(c_matrix)


class ScalpelTime:
    def __init__(self, filename, impute, clf_desc):
        self.df = pd.read_pickle(filename)

        # Restrict to early MSH
        separation_date = pd.to_datetime(Config.cutoff_date)
        self.df = self.df.query('Admit_Date < @separation_date and Last_Facility == "MSH"')
        self.df = self.df.drop(['Admit_Date', 'Last_Facility'], axis=1)

        # Drop columns
        self.keep_cols = self.df.columns[
            (self.df.count() / self.df.shape[0]) > Config.values_present]
        self.df = self.df[self.keep_cols]
        print('Min present:', (self.df.count() / len(self.df)).min())

        # Scale
        scale = self.df.shape[0] // self.df['DIALYSIS'].sum()
        print('Scale:', scale)

        self.impute = impute

        self.outcome = filename.split('/')[1].split('_')[0]
        self.reporting_oc = filename.split('/')[1].replace('.pickle', '')

        # IMPORTANT
        parameters_xgb = {
            'n_estimators': [40, 60, 80, 100],
            'learning_rate': [0.05, 0.10, 0.15, 0.20],
            'max_depth': [3, 4, 5, 6, 8, 10, 12, 15],
            'min_child_weight': [1, 3, 5, 7],
            'gamma' : [0.0, 0.1, 0.2, 0.3, 0.4],
            'colsample_bytree': [0.3, 0.4, 0.5, 0.7]
        }
        parameters_lr = {
            'C': np.logspace(-3, 3, 7)
        }

        clf = XGBClassifier(
            n_jobs=-1, gpu_id=0,
            max_delta_step=1, scale_pos_weight=scale)
        clf = LogisticRegression(penalty='l1', solver='liblinear')
        #clf = LogisticRegression(n_jobs=-1)

        dict_parameters = {
            'xgb': parameters_xgb,
            'lr': parameters_lr,
            'LASSO': parameters_lr
        }
        parameters = dict_parameters[clf_desc]

        self.rgs = RandomizedSearchCV(
            clf,
            parameters,
            n_iter=Config.gs_iterations,
            scoring='f1',
            n_jobs=os.cpu_count() - 1,
            cv=Config.validation_folds,
            random_state=Config.random_state,
            return_train_score=True,
            verbose=1
        )

        self.metrics = []

    def imputer(self, df):
        imputed = KNNImputer(
            n_neighbors=5, weights="uniform").fit_transform(df)
        df_imputed = pd.DataFrame(
            imputed, columns=df.columns, index=df.index)

        return df_imputed

    def exploratory_laparotomy(self):
        X = self.df.drop(self.outcome, axis=1)
        y = self.df[[self.outcome]].values.ravel()

        if self.impute:
            X = self.imputer(X)

        self.rgs.fit(X, y)
        improved_score = self.rgs.best_score_

        metrics = [
            self.reporting_oc,
            improved_score,
            self.rgs.best_params_
        ]

        return metrics
