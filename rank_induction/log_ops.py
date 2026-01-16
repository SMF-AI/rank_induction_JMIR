import os
import logging
import datetime
from logging.handlers import TimedRotatingFileHandler


import mlflow
from matplotlib import pyplot as plt

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT_DIR, "logs")

TRACKING_URI = "http://192.168.10.112:5000/"
EXP_NAME = "rank_induction"


def get_experiment(experiment_name: str = EXP_NAME):
    mlflow.set_tracking_uri(TRACKING_URI)

    client = mlflow.tracking.MlflowClient(TRACKING_URI)
    experiment = client.get_experiment_by_name(experiment_name)

    if not experiment:
        client.create_experiment(experiment_name)
        return client.get_experiment_by_name(experiment_name)

    return experiment


 


def save_and_log_figure(filename: str) -> None:
    """그려진 figure을 MLflow내에 figure을 등록함

    Note:
        mlflow내에 figure을 바로 넣는 메서드가 없음. disk에 저장 필요.

    Args:
        filename (str): filename

    Example:
        >>> from seedp.metrics import plot_auroc
        >>> plot_auroc(metrics.labels, metrics.probs)
        >>> save_and_log_figure("auroc.png")

    """
    plt.savefig(filename)
    mlflow.log_artifact(filename)
    os.remove(filename)
    plt.clf()

    return
