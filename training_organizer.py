#!/bin/python

import os

import pandas as pd
import numpy as np
import scipy.stats as st

from config import Config
from train_test_model import HammerTime


class Organizer:
    def __init__(self, filename):
        os.makedirs(Config.results_dir, exist_ok=True)
        self.df = pd.read_pickle(filename)

        # Rename features - Hyphens cause issues
        columns = [
            i.replace(' ', '').replace('-', '_')
            for i in self.df.columns]
        self.df.columns = columns

        # This is finicky
        self.outcome = filename.split('/')[1].split('_')[0]
        self.day = filename.split('/')[1].split('_')[1].replace('.pickle', '')

        self.df_msh = None
        self.df_oh = None

    def train_test_validate_model(self):
        # Subdivide df_train_test on the basis of Last_Facility
        # Cross validate on df_msh / Test on df_oh
        self.df_msh = self.df.query('Last_Facility == "MSH"').\
            drop(['Admit_Date', 'Last_Facility'], axis=1)
        self.df_oh = self.df.query('Last_Facility != "MSH"').\
            drop(['Admit_Date', 'Last_Facility'], axis=1)

        # Show sizes of each cohort
        print(
            'MSH:', self.df_msh.shape[0],
            'OH:', self.df_oh.shape[0]
        )

        # Cross validation is automatic
        print('Cross validating...')
        hammer_time = HammerTime(self.df_msh, self.outcome, self.day)

        # Save thresholds - have to be rounded up
        pd.to_pickle(
            hammer_time.dict_thresholds,
            f'Results/thresh_{self.outcome}_{self.day}',
            protocol=4)

        # Additional testing and validation have to be called
        print('Testing and validating...')
        hammer_time.test_validate(self.df_oh, 'test')

        self.aggregate_metrics(hammer_time.dict_metrics)

    def calc_sens_spec(self, cm):
        sens = cm[0, 0] / (cm[0, 0] + cm[0, 1])
        spec = cm[1, 1] / (cm[1, 1] + cm[1, 0])
        return sens, spec

    def calc_mean_ci(self, arr):
        mean = np.mean(arr)
        ci = st.t.interval(0.95, len(arr)-1, loc=mean, scale=st.sem(arr))
        return f'{mean.round(3)} ({round(ci[0], 3)} - {round(ci[1], 3)})'

    def aggregate_metrics(self, dict_metrics):

        # Save raw metrics
        outcome = f'{self.outcome}_{self.day}'
        outfile = f'raw_{outcome}.pickle'
        pd.to_pickle(dict_metrics, os.path.join(Config.results_dir, outfile))

        # df from here
        all_metrics = []

        # This is left redundant on purpose
        # Cross val results
        for clf_desc in Config.clf_descriptions:

            desc = 'MSH > MSH'
            patients = self.df_msh.shape[0]
            outcomes = self.df_msh[self.outcome].sum()
            outcome_perc = outcomes / patients

            acc_arr = dict_metrics['cross_val'][clf_desc]['acc']
            acc = self.calc_mean_ci(acc_arr)

            auroc_arr = dict_metrics['cross_val'][clf_desc]['auroc']
            auroc = self.calc_mean_ci(auroc_arr)

            auprc_arr = dict_metrics['cross_val'][clf_desc]['auprc']
            auprc = self.calc_mean_ci(auprc_arr)

            f1s_arr = dict_metrics['cross_val'][clf_desc]['f1s']
            f1s = self.calc_mean_ci(f1s_arr)

            sens_arr = dict_metrics['cross_val'][clf_desc]['sens']
            sens = self.calc_mean_ci(sens_arr)

            spec_arr = dict_metrics['cross_val'][clf_desc]['spec']
            spec = self.calc_mean_ci(spec_arr)

            metrics = [
                outcome, desc, clf_desc, patients, outcomes, outcome_perc,
                acc, auroc, auprc, f1s, sens, spec]
            all_metrics.append(metrics)

            # Better descriptions
            dict_desc = {
                'test': ['MSH > OH', self.df_oh],
            }

        # Test / validate
        for key in dict_desc:
            for clf_desc in Config.clf_descriptions:

                desc = dict_desc[key][0]
                patients = dict_desc[key][1].shape[0]
                outcomes = dict_desc[key][1][self.outcome].sum()
                outcome_perc = outcomes / patients
                acc = round(dict_metrics[key][clf_desc]['acc'][0], 3)
                auroc = round(dict_metrics[key][clf_desc]['auroc'][0], 3)
                auprc = round(dict_metrics[key][clf_desc]['auprc'][0], 3)
                f1s = round(dict_metrics[key][clf_desc]['f1s'][0], 3)

                cm = dict_metrics[key][clf_desc]['c_matrix'][0]
                sens, spec = self.calc_sens_spec(cm)

                metrics = [
                    outcome, desc, clf_desc, patients, outcomes, outcome_perc, acc,
                    auroc, auprc, f1s, sens, spec]
                all_metrics.append(metrics)

        df_metrics = pd.DataFrame(all_metrics)
        df_metrics = df_metrics.round(3)
        df_metrics.columns = [
            'OUTCOME', 'OPERATION', 'CLASSIFIER', 'PATIENTS', 'POSITIVE OUTCOMES', 'OUTCOME PERC',
            'ACCURACY', 'AUROC', 'AUPRC', 'F1S', 'SENS', 'SPEC']

        df_metrics.to_pickle(
            f'{Config.results_dir}/agg_{outcome}.pickle', protocol=4)

    def start(self):
        self.train_test_validate_model()
