from .easy import easy_task
from .medium import medium_task
from .hard import hard_task

def get_task(task_name: str):
    tasks = {
        "easy": easy_task,
        "medium": medium_task,
        "hard": hard_task
    }
    if task_name not in tasks:
        raise ValueError(f"Task {task_name} not found.")
    return tasks[task_name]()
