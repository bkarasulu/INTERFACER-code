"""Execute the EMT tutorial notebooks (Tutorials 1 and 6) in a copy of their folders (needs nbclient and ipykernel)."""
import os
import shutil

import pytest

nbformat = pytest.importorskip("nbformat")
nbclient = pytest.importorskip("nbclient")
pytest.importorskip("ipykernel")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.parametrize("tutorial", ["01_Cu-Ni_EMT", "06_grain_boundaries_trilayer"])
def test_tutorial_notebook_runs(tutorial, tmp_path):
    if tutorial.startswith("06") and not os.environ.get("INTERFACER_SLOW_TESTS"):
        pytest.skip("about a minute; set INTERFACER_SLOW_TESTS=1")
    work = tmp_path / tutorial
    shutil.copytree(os.path.join(ROOT, "Tutorials", tutorial), work,
                    ignore=shutil.ignore_patterns("OUTPUT*", "*.out", "*.log"))
    nb = nbformat.read(work / "analysis.ipynb", as_version=4)
    nbclient.NotebookClient(nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(work)}}).execute()
    errors = [o for c in nb.cells for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert not errors
    assert (work / "OUTPUT" / "results.json").exists()
