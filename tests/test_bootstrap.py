"""Bootstrap acceptance checks; use python -m unittest discover -s tests -v."""
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from mlrf.config import load_project_config
from mlrf.data.contract import load_ds04_contract, validate_raw_columns


class ProjectBootstrapTests(unittest.TestCase):
    def test_config_target_seed_and_split(self):
        config = load_project_config()
        self.assertEqual(config["dataset"]["repository"], "imbalet/used_cars")
        self.assertEqual(config["dataset"]["target"], "price")
        self.assertEqual(config["split"]["random_seed"], 42)
        self.assertEqual(config["split"]["status"], "provisional_pending_D13")

    def test_ds04_column_contract(self):
        schema = load_ds04_contract()
        cols = schema["numeric_columns"] + schema["categorical_columns"]
        self.assertEqual(len(cols), 21)
        self.assertIn("price", cols)
        validate_raw_columns(cols)
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_raw_columns([column for column in cols if column != "price"])

    def test_kaggle_notebook_is_valid_json(self):
        path = ROOT / "notebooks" / "00_environment_check.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        self.assertTrue(any(cell["cell_type"] == "code" for cell in notebook["cells"]))
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), "<notebook cell>", "exec")


if __name__ == "__main__":
    unittest.main()
