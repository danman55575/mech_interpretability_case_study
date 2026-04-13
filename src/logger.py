import os
from dotenv import load_dotenv
from clearml import Task
from src.config import cfg


def init_clearml(
    task_name: str, project_name: str = "Mechanistic_Interpretability_SAE"
) -> Task:
    """
    Initializes a ClearML task and logs the global configuration.
    Requires .env file with CLEARML credentials.
    """
    # Load environment variables from .env file
    load_dotenv()

    # Ensure keys are present
    if not os.getenv("CLEARML_API_ACCESS_KEY"):
        print(
            "[WARNING] ClearML credentials not found in environment. Running in offline/debug mode."
        )
        Task.set_offline(offline_mode=True)

    # Initialize task
    task = Task.init(project_name=project_name, task_name=task_name)

    # Log hyperparameters
    task.connect(
        {
            "model": cfg.model.__dict__,
            "sae": cfg.sae.__dict__,
            "train": cfg.train.__dict__,
        },
        name="Experiment_Config",
    )
    return task
