"""
ClearML experiment tracking integration.
"""

from __future__ import annotations

import os
from typing import Optional

from dotenv import load_dotenv

from src.config import ExperimentConfig


def init_clearml(
    cfg: ExperimentConfig,
    task_name: str,
    project_name: str = "Mechanistic_Interpretability_SAE",
) -> Optional[object]:
    """
    Initializes a ClearML task and logs the experiment configuration.

    Falls back to offline mode if credentials are not available.

    Args:
        cfg: Experiment configuration to log.
        task_name: Name for the ClearML task.
        project_name: ClearML project name.

    Returns:
        ClearML Task object, or None if ClearML is unavailable.
    """
    load_dotenv()

    try:
        from clearml import Task
    except ImportError:
        print("[WARNING] ClearML not installed. Skipping experiment tracking.")
        return None

    if not os.getenv("CLEARML_API_ACCESS_KEY"):
        print("[WARNING] ClearML credentials not found. Running in offline mode.")
        Task.set_offline(offline_mode=True)

    task = Task.init(project_name=project_name, task_name=task_name)
    task.connect(cfg.to_dict(), name="Experiment_Config")

    return task


def get_logger(task: Optional[object]):
    """
    Returns a ClearML Logger from the given task, or a no-op fallback.

    Args:
        task: ClearML Task object, or None.

    Returns:
        ClearML Logger or NullLogger.
    """
    if task is not None:
        from clearml import Logger
        return Logger.current_logger()
    return NullLogger()


class NullLogger:
    """
    Logger that silently ignores all logging calls.

    Used as a drop-in replacement when ClearML is unavailable.
    """

    def report_scalar(self, *args, **kwargs) -> None:
        pass

    def report_single_value(self, *args, **kwargs) -> None:
        pass

    def report_text(self, *args, **kwargs) -> None:
        pass
