"""Tests proving our custom forest is not a one-line sklearn RandomForest call."""
import sys
import unittest
from pathlib import Path
import numpy as np
from sklearn.datasets import make_regression
from sklearn.exceptions import NotFittedError
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mlrf.models import CustomRandomForestRegressor, build_decision_tree, build_dummy
from mlrf.training import build_model, model_search_space
from mlrf.evaluation import regression_metrics


class ModelsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X, cls.y = make_regression(n_samples=120, n_features=5, noise=8.0, random_state=42)

    def test_dummy_and_tree(self):
        for m in (build_dummy(), build_decision_tree(max_depth=4), build_model('decision_tree', {'max_depth': 3})):
            pred = m.fit(self.X, self.y).predict(self.X)
            self.assertEqual(pred.shape, (120,))
            self.assertTrue(np.isfinite(pred).all())
        self.assertIn('max_depth', model_search_space('decision_tree'))

    def test_forest_training_aggregation_and_oob(self):
        m = CustomRandomForestRegressor(n_estimators=12, max_depth=5, max_features=.8,
                                       oob_score=True, random_state=42)
        with self.assertRaises((NotFittedError, AttributeError)):
            m.predict(self.X)
        m.fit(self.X, self.y)
        self.assertEqual(len(m.estimators_), 12)
        self.assertEqual(len(m.bootstrap_indices_), 12)
        self.assertTrue(np.any(m.oob_counts_ > 0))
        self.assertTrue(np.isfinite(m.oob_score_))
        expected = np.mean([t.predict(self.X) for t in m.estimators_], axis=0)
        np.testing.assert_allclose(m.predict(self.X), expected, atol=1e-10)
        self.assertEqual(len(m.feature_importances_), 5)
        self.assertAlmostEqual(m.feature_importances_.sum(), 1, places=5)

    def test_reproducibility_and_config_guards(self):
        args = dict(n_estimators=10, random_state=42, max_depth=3)
        first = CustomRandomForestRegressor(**args).fit(self.X, self.y).predict(self.X)
        second = CustomRandomForestRegressor(**args).fit(self.X, self.y).predict(self.X)
        np.testing.assert_array_equal(first, second)
        with self.assertRaises(ValueError):
            CustomRandomForestRegressor(oob_score=True, bootstrap=False).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            CustomRandomForestRegressor(n_estimators=0).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            CustomRandomForestRegressor(max_samples=200).fit(self.X, self.y)
        with self.assertRaises(ValueError):
            CustomRandomForestRegressor(n_estimators=2).fit(self.X, self.y).predict(self.X[:, :2])

    def test_metrics(self):
        metrics = regression_metrics([10, 20, 30], [12, 18, 29])
        self.assertAlmostEqual(metrics['mae'], 5 / 3)
        self.assertGreater(metrics['rmse'], metrics['mae'])
        with self.assertRaises(ValueError):
            regression_metrics([1, 2], [1])

if __name__ == '__main__':
    unittest.main()
