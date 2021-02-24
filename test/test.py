import unittest

import pandas as pd
from sklearn.metrics import max_error as sklearn_max_error
from sklearn.metrics import mean_absolute_error as sklearn_mae
from sklearn.metrics import mean_squared_error as sklearn_mse
from sklearn.metrics import median_absolute_error as sklearn_medae
from sklearn.metrics import mean_absolute_percentage_error as sklearn_mape

from app.algo import check, compute_local_prediction_error, aggregate_prediction_errors, mae, rmse, max_error, mse, \
    medae


class TestRegressionEvaluation(unittest.TestCase):
    def setUp(self):
        y_proba = pd.read_csv("y_proba.csv")
        y_test = pd.read_csv("y_test.csv")

        y_proba1 = pd.read_csv("client1/y_proba.csv")
        y_proba2 = pd.read_csv("client2/y_proba.csv")

        y_test1 = pd.read_csv("client1/y_test.csv")
        y_test2 = pd.read_csv("client2/y_test.csv")

        self.y_test, self.y_proba = check(y_test, y_proba)
        self.y_test1, self.y_proba1 = check(y_test1, y_proba1)
        self.y_test2, self.y_proba2 = check(y_test2, y_proba2)

        self.pred_errors_central = compute_local_prediction_error(self.y_test, self.y_proba)

        pred_errors1 = compute_local_prediction_error(self.y_test1, self.y_proba1)
        pred_errors2 = compute_local_prediction_error(self.y_test2, self.y_proba2)
        self.pred_errors_global = aggregate_prediction_errors([pred_errors1, pred_errors2])

    def test_prediction_errors(self):
        for i in range(len(self.pred_errors_central)):
            self.assertEqual(self.pred_errors_central[i], self.pred_errors_global[i])

    def test_mae(self):
        mae_sklearn = sklearn_mae(self.y_test, self.y_proba)
        print(mae_sklearn)
        mae_central = mae(self.pred_errors_central)
        mae_global = mae(self.pred_errors_global)
        print(mae_global)

        self.assertEqual(mae_sklearn, mae_central)
        self.assertEqual(mae_central, mae_global)

    def test_rmse(self):
        rmse_sklearn = sklearn_mse(self.y_test, self.y_proba, squared=False)
        rmse_central = rmse(self.pred_errors_central)
        rmse_global = rmse(self.pred_errors_global)

        self.assertEqual(rmse_sklearn, rmse_central)
        self.assertEqual(rmse_central, rmse_global)

    def test_mse(self):
        mse_sklearn = sklearn_mse(self.y_test, self.y_proba, squared=True)
        mse_central = mse(self.pred_errors_central)
        mse_global = mse(self.pred_errors_global)

        self.assertEqual(mse_sklearn, mse_central)
        self.assertEqual(mse_central, mse_global)

    def test_max_error(self):
        max_error_sklearn = sklearn_max_error(self.y_test, self.y_proba)
        max_error_central = max_error(self.pred_errors_central)
        max_error_global = max_error(self.pred_errors_global)

        self.assertEqual(max_error_sklearn, max_error_central)
        self.assertEqual(max_error_central, max_error_global)

    def test_msle(self):
        medae_sklearn = sklearn_medae(self.y_test, self.y_proba)
        medae_central = medae(self.pred_errors_central)
        medae_global = medae(self.pred_errors_global)

        self.assertEqual(medae_sklearn, medae_central)
        self.assertEqual(medae_central, medae_global)


if __name__ == "__main__":
    unittest.main()
