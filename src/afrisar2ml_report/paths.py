import platform
from pathlib import Path

from afrisar2ml_report import config


def get_project_folder():
    """Path to the project folder (includes dataset, cached processing steps, etc.)."""
    if platform.system() == "Windows":
        polinsar_folder = Path("//hr-fs/HR_Data/Pol-InSAR_InfoRetrieval/")
    else:
        polinsar_folder = Path("/hrdss/HR_Data/Pol-InSAR_InfoRetrieval/")
    project_folder = polinsar_folder / "01_projects/AFRISAR_ML"
    project_folder.mkdir(parents=True, exist_ok=True)
    return project_folder


def get_dataset_path(site):
    zarr_path = get_project_folder() / "dataset" / f"AfriSAR-2-ML_{site}_v{config.DS_VERSION}"
    return zarr_path


def get_cache_folder():
    """Path to intermediate benchmark results."""
    folder = get_project_folder() / "cache_report"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def get_report_figure_folder():
    folder = get_project_folder() / "figures_report"
    folder.mkdir(parents=True, exist_ok=True)
    return folder
