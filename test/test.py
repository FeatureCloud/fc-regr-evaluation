import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from sklearn.metrics import roc_auc_score

from app.algo import check, aggregate_confusion_matrices, compute_threshold_conf_matrices, \
    compute_min_max_score, agg_compute_thresholds, compute_roc_parameters, roc_plot, compute_roc_auc, find_nearest, \
    create_score_df


class TestROC(unittest.TestCase):
    def setUp(self):
        y_proba = pd.read_csv("y_proba.csv")
        y_test = pd.read_csv("y_test.csv")

        y_proba1 = y_proba.iloc[:150, :]
        y_proba2 = y_proba.iloc[150:, :]

        y_test1 = y_test.iloc[:150, :]
        y_test2 = y_test.iloc[150:, :]

        y_test, y_proba = check(y_test, y_proba)
        y_test1, y_proba1 = check(y_test1, y_proba1)
        y_test2, y_proba2 = check(y_test2, y_proba2)

        min, max = compute_min_max_score(y_proba)
        min1, max1 = compute_min_max_score(y_proba1)
        min2, max2 = compute_min_max_score(y_proba2)
        self.thresholds_central = agg_compute_thresholds([[min, max]])
        self.thresholds_global = agg_compute_thresholds([[min1, max1], [min2, max2]])

        self.confusion_matrices_central = compute_threshold_conf_matrices(y_test, y_proba, self.thresholds_central)
        confs1 = compute_threshold_conf_matrices(y_test1, y_proba1, self.thresholds_global)
        confs2 = compute_threshold_conf_matrices(y_test2, y_proba2, self.thresholds_global)
        self.confusion_matrices_global = aggregate_confusion_matrices([confs1, confs2])
        idx = find_nearest(self.thresholds_global, 0.5)
        self.confusion_matrix_global = self.confusion_matrices_global[idx]
        self.roc_params_central = compute_roc_parameters(self.confusion_matrices_central, self.thresholds_central)
        self.roc_params_global = compute_roc_parameters(self.confusion_matrices_global, self.thresholds_global)

        plot_central, self.df_central = roc_plot(self.roc_params_central["FPR"], self.roc_params_central["TPR"],
                                                 self.roc_params_central["THR"])

        plot_global, self.df_global = roc_plot(self.roc_params_global["FPR"], self.roc_params_global["TPR"],
                                               self.roc_params_global["THR"])


        self.auc_central = roc_auc_score(y_test, y_proba)
        self.auc_global = compute_roc_auc(self.roc_params_global["FPR"], self.roc_params_global["TPR"])
        df_scores = create_score_df(self.confusion_matrix_global, self.auc_global)

    def test_thresholds(self):
        for i in range(len(self.thresholds_central)):
            self.assertEqual(self.thresholds_central[i], self.thresholds_global[i])

    def test_confs(self):
        for i in range(len(self.confusion_matrices_central)):
            self.assertDictEqual(self.confusion_matrices_central[i], self.confusion_matrices_global[i])

    def test_roc_params(self):
        for key in self.roc_params_central.keys():
            for i in range(len(self.roc_params_central[key])):
                self.assertEqual(self.roc_params_central[key][i], self.roc_params_global[key][i])

    def test_frames(self):
        assert_frame_equal(self.df_central, self.df_global)

    def test_auc(self):
        self.assertEqual(self.auc_global, self.auc_central)


if __name__ == "__main__":
    unittest.main()
