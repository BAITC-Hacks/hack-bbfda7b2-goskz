import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import starter


class StarterTests(unittest.TestCase):
    def test_validate_startup_accepts_complete_project_layout(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "app.py").touch()
            data_dir = root / "parquet"
            data_dir.mkdir()
            for name in ("edges.parquet", "nodes.parquet", "transactions.parquet"):
                (data_dir / name).touch()
            with patch("starter.importlib.util.find_spec", return_value=object()):
                starter.validate_startup(root)

    def test_launch_uses_current_python_and_project_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch("starter.subprocess.run") as run:
                starter.launch_streamlit(root)
            run.assert_called_once_with(
                [sys.executable, "-m", "streamlit", "run", str(root / "app.py")],
                check=True,
                cwd=str(root),
            )


if __name__ == "__main__":
    unittest.main()
