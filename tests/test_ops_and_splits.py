"""TV3 data integrity + artifact smoke tests using synthetic data only."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mlrf.testing import assert_target_absent, assert_disjoint_ids, split_indices, inspect_split
from mlrf.training import build_model
from mlrf.ops import save_run, save_trusted_model, load_trusted_model
from mlrf.evaluation import regression_metrics


class SafetyTests(unittest.TestCase):
    def test_disjoint_group_and_target_guards(self):
        tr, de, te = split_indices(100, random_state=42)
        self.assertEqual(sum(inspect_split(tr, de, te).values()), 100)
        with self.assertRaises(ValueError):
            assert_disjoint_ids([0, 1], [1, 2], [3])
        with self.assertRaises(ValueError):
            assert_target_absent(pd.DataFrame({'price': [3], 'year':[2020]}))
        groups = np.repeat(np.arange(40), 3)
        tr, de, te = split_indices(len(groups), groups=groups)
        self.assertFalse(set(groups[tr]) & set(groups[te]))
        self.assertFalse(set(groups[tr]) & set(groups[de]))
        self.assertFalse(set(groups[de]) & set(groups[te]))

    def test_end_to_end_train_only_preprocessing_and_artifacts(self):
        rng = np.random.default_rng(42)
        data = pd.DataFrame({'year': rng.integers(2000, 2025, 90),
                             'km': rng.integers(2000, 180000, 90).astype(float),
                             'brand': np.where(np.arange(90) % 2, 'A', 'B')})
        data.loc[5, 'km'] = np.nan
        y = np.maximum(1000, 40000 + 1000 * (data.year - 2020) - .08 * data.km.fillna(0) + rng.normal(0, 400, 90))
        tr, de, te = split_indices(90)
        trans = ColumnTransformer([('num', SimpleImputer(strategy='median'), ['year', 'km']),
                                   ('cat', OneHotEncoder(handle_unknown='ignore'), ['brand'])])
        # Entire preprocessing is fitted on training rows only; untouched test set.
        model = Pipeline([('pre', trans), ('reg', build_model('custom_random_forest',
                                    {'n_estimators': 8, 'max_depth': 5, 'random_state': 42}))])
        model.fit(data.iloc[tr], y.iloc[tr])
        predictions = model.predict(data.iloc[de])
        scores = regression_metrics(y.iloc[de], predictions)
        self.assertTrue(np.isfinite(list(scores.values())).all())
        self.assertEqual(len(predictions), len(de))
        with tempfile.TemporaryDirectory() as tmp:
            path = save_run(tmp, 'rf_smoke_001', config={'seed': 42}, metrics=scores,
                            dataset={'source': 'synthetic_test_only'},
                            predictions={'y_pred': predictions})
            self.assertTrue((path / 'manifest.json').exists())
            self.assertEqual(json.loads((path / 'manifest.json').read_text())['run_id'], 'rf_smoke_001')
            save_trusted_model(model, path / 'pipeline.joblib')
            restored = load_trusted_model(path / 'pipeline.joblib')
            np.testing.assert_array_equal(restored.predict(data.iloc[te]), model.predict(data.iloc[te]))
            with self.assertRaises(FileExistsError):
                save_run(tmp, 'rf_smoke_001', config={}, metrics={}, dataset={})
            with self.assertRaises(ValueError):
                save_run(tmp, '../bad', config={}, metrics={}, dataset={})

if __name__ == '__main__':
    unittest.main()
