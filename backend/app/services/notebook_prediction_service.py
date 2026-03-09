import json
import os
from pathlib import Path

import nbformat
import pandas as pd
from nbclient import NotebookClient


class NotebookPredictionService:
    def __init__(self) -> None:
        notebook_path = os.getenv("NOTEBOOK_PREDICT_PATH", "ml/predict.ipynb")
        self.notebook_path = Path(notebook_path).resolve()
        self.kernel_name = os.getenv("NOTEBOOK_KERNEL", "python3")

    def _load_notebook(self):
        if not self.notebook_path.exists():
            raise FileNotFoundError(f"Notebook not found at {self.notebook_path}")
        with open(self.notebook_path, "r", encoding="utf-8") as fh:
            return nbformat.read(fh, as_version=4)

    def _inject_period(self, notebook, predict_year: int, predict_month: int):
        injected = False
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue

            lines = cell.source.splitlines()
            replaced_any = False
            updated_lines = []
            for line in lines:
                stripped = line.strip()
                if stripped.startswith("PREDICT_YEAR"):
                    updated_lines.append(f"PREDICT_YEAR  = {predict_year}")
                    replaced_any = True
                elif stripped.startswith("PREDICT_MONTH"):
                    updated_lines.append(f"PREDICT_MONTH = {predict_month}")
                    replaced_any = True
                else:
                    updated_lines.append(line)

            if replaced_any:
                cell.source = "\n".join(updated_lines)
                injected = True
                break

        if not injected:
            raise ValueError(
                "Could not find PREDICT_YEAR/PREDICT_MONTH cell in notebook. "
                "Please keep the input cell in ml/predict.ipynb."
            )

    def predict_with_notebook(self, predict_year: int, predict_month: int) -> pd.DataFrame:
        if not 1 <= predict_month <= 12:
            raise ValueError("predict_month must be between 1 and 12")

        notebook = self._load_notebook()
        self._inject_period(notebook, predict_year, predict_month)

        client = NotebookClient(
            notebook,
            timeout=1800,
            kernel_name=self.kernel_name,
            resources={"metadata": {"path": str(self.notebook_path.parent)}},
        )
        client.execute()

        expected_output = (
            self.notebook_path.parent
            / "outputs"
            / f"prediction_{predict_year}_{predict_month:02d}.xlsx"
        )
        if not expected_output.exists():
            raise FileNotFoundError(
                "Notebook executed but prediction output file was not found at "
                f"{expected_output}."
            )

        df = pd.read_excel(expected_output)
        if df.empty:
            raise ValueError("Prediction output is empty.")

        # Normalize headers for API mapping.
        df.columns = [str(col).strip().lower() for col in df.columns]
        return df
