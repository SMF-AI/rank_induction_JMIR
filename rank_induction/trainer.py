import math
import uuid
import logging
from copy import deepcopy
from typing import Tuple, Literal
from abc import ABC, abstractmethod

import mlflow
import torch
import numpy as np
from torch.utils.data import DataLoader
from progress.bar import Bar

from rank_induction.data_models import BinaryLabels
from rank_induction.metrics import (
    AverageMeter,
    MetricsMeter,
    plot_auroc,
    plot_prauc,
    plot_confusion_matrix,
)
from rank_induction.log_ops import save_and_log_figure


class BaseTrainer(ABC):
    @abstractmethod
    def run_epoch(self):
        pass


class BinaryClassifierTrainer(BaseTrainer):
    """이진분류기 학습기"""

    def __init__(
        self,
        model: torch.nn.modules.Module,
        loss: torch.nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer = None,
        logger: logging.Logger = None,
        device: str = "cuda",
    ):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.logger = (
            logging.Logger("BinaryClassifierTrainer") if logger is None else logger
        )
        self.device = device

    def make_bar_sentence(
        self,
        phase: str,
        epoch: int,
        total_step: int,
        step: int,
        eta: str,
        total_loss: float,
        metrics_meter: MetricsMeter,
    ) -> str:
        """ProgressBar의 stdout의 string을 생성하여 반환

        Args:
            phase (str): Epoch의 phase
            epoch (int): epoch
            total_step (int): total steps for one epoch
            step (int): Step (in a epoch)
            eta (str): Estimated Time of Arrival
            loss (float): loss
            metrics_meter: MetricsMeters class

        Returns:
            str: progressbar sentence

        """
        sentence = (
            f"{phase} | EPOCH {epoch}: [{step}/{total_step}] | "
            f"eta:{eta} | total_loss: {round(total_loss, 5)} | "
        )

        sentence += " | ".join(
            [f"{k}: {round(v, 5)}" for k, v in metrics_meter.to_dict().items()]
        )

        return sentence

    def run_epoch(
        self,
        dataloader: DataLoader,
        phase: Literal["train", "val", "test"],
        current_epoch: int,
        accumulation_steps: int = 1,
        verbose: bool = True,
    ) -> Tuple[AverageMeter, MetricsMeter]:
        """1회 Epoch을 각 페이즈(train, val, test)에 따라, 예측/훈련함

        Args:
            dataloader (DataLoader): DataLoader
            phase (str): train, val, test
                - train인 경우 파라미터를 학습, val,test인 경우는 torch.no_grad적용
            current_epoch (int): 현재 Epoch
            accumulation_steps (int): gradient accumulation steps

        Returns:
            Tuple[AverageMeter, MetricsMeter]
                - AverageMeter: 평균 Empirical loss
                - MetricsMeter: 평균 Metrics

        Example:
            >>> trainer = BinaryClassifierTrainer(
                    model=model,
                    loss=nn.BCEWithLogitsLoss(),
                    optimizer=optimizer,
                )
            >>> loss, metrics = trainer.run_epoch(dataloader, phase="train", epoch=1)
            >>> mlflow.log_metric("train_loss", loss.avg, step=current_epoch)
            >>> mlflow.log_metrics(metrics.to_dict(), step=current_epoch)
        """
        total_step = len(dataloader)
        if verbose:
            bar = Bar(max=total_step, check_tty=False)

        loss_meter = AverageMeter("loss")
        metrics_meter = MetricsMeter(name=phase, accuracy_threshold=0.5)
        for step, batch in enumerate(dataloader, start=1):
            xs, ys = batch  # (1, K, 512,) (1,)
            xs = xs.to(self.device)
            ys = ys.to(self.device)

            if phase == "train":
                self.model.train()
                logits, _ = self.model(xs)  # (1, 1)
            else:
                self.model.eval()
                with torch.no_grad():
                    logits, _ = self.model(xs)

            ys = ys.view(-1, 1)
            logits = logits.view(-1, 1)
            loss = self.loss(logits, ys)
            loss = loss / accumulation_steps

            if phase == "train":
                loss.backward()
                if step % accumulation_steps == 0 or (step == total_step):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # metric
            loss_meter.update(loss.item() * accumulation_steps, len(ys))
            model_confidence: float = torch.sigmoid(logits).item()
            metrics_meter.update(
                [model_confidence],
                [ys.item()],
            )

            if verbose:
                bar.suffix = self.make_bar_sentence(
                    phase=phase,
                    epoch=current_epoch,
                    step=step,
                    total_step=total_step,
                    eta=bar.eta,
                    total_loss=loss_meter.avg,
                    metrics_meter=metrics_meter,
                )
                bar.next()

        if verbose:
            bar.finish()

        return (loss_meter, metrics_meter)

    def train(
        self,
        train_dataloader,
        val_dataloader=None,
        n_epochs: int = 100,
        max_patiences: int = 10,
        minimal_earlystop_epoch: int = 50,
        accumulation_steps: int = 1,
        use_mlflow: bool = True,
        verbose: bool = True,
    ):

        best_params = self.model.state_dict()
        best_loss = math.inf
        patience = 0
        for current_epoch in range(1, n_epochs + 1):
            train_loss, train_metrics = self.run_epoch(
                train_dataloader,
                phase="train",
                current_epoch=current_epoch,
                accumulation_steps=accumulation_steps,
                verbose=verbose,
            )
            if use_mlflow:
                mlflow.log_metric("train_loss", train_loss.avg, step=current_epoch)
                mlflow.log_metrics(train_metrics.to_dict(), step=current_epoch)

            if val_dataloader is None:
                continue

            val_loss, val_metrics = self.run_epoch(
                val_dataloader,
                phase="val",
                current_epoch=current_epoch,
                verbose=verbose,
            )
            if use_mlflow:
                mlflow.log_metric("val_loss", val_loss.avg, step=current_epoch)
                mlflow.log_metrics(val_metrics.to_dict(), step=current_epoch)

            if best_loss > val_loss.avg:
                best_loss = val_loss.avg
                patience = 0
                best_params = deepcopy(self.model.state_dict())
            else:
                patience += 1
                if (
                    patience >= max_patiences
                    and current_epoch >= minimal_earlystop_epoch
                ):
                    break

        self.model.load_state_dict(best_params)
        return

    def test(
        self, test_loader: DataLoader, use_mlflow: bool = True, verbose: bool = True
    ):
        loss, metrics = self.run_epoch(
            test_loader, phase="test", current_epoch=0, verbose=verbose
        )
        if use_mlflow:
            mlflow.log_metric("test_loss", loss.avg, step=0)
            mlflow.log_metrics(metrics.to_dict(), step=0)

            unique_id = uuid.uuid4()
            plot_auroc(metrics.labels, metrics.probs)
            save_and_log_figure(f"auroc_{unique_id}.png")

            plot_prauc(metrics.labels, metrics.probs)
            save_and_log_figure(f"auprc_{unique_id}.png")

            plot_confusion_matrix(
                y_true=metrics.labels,
                y_pred=[p > 0.5 for p in metrics.probs],
                labels=[label.name for label in BinaryLabels],
            )
            save_and_log_figure(f"confusion_matrix_{unique_id}.png")


class MixedSupervisionTrainer(BinaryClassifierTrainer):
    """다중분류기 학습기"""

    def __init__(
        self,
        model: torch.nn.modules.Module,
        loss: torch.nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer = None,
        logger: logging.Logger = None,
        device: str = "cuda",
    ):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.logger = (
            logging.Logger("MulticlassClassifierTrainer") if logger is None else logger
        )
        self.device = device

    def run_epoch(
        self,
        dataloader: DataLoader,
        phase: Literal["train", "val", "test"],
        current_epoch: int,
        accumulation_steps: int = 1,
        verbose: bool = True,
    ) -> Tuple[AverageMeter, MetricsMeter]:
        """1회 Epoch을 각 페이즈(train, val, test)에 따라, 예측/훈련함

        Args:
            dataloader (DataLoader): DataLoader
            phase (str): train, val, test
                - train인 경우 파라미터를 학습, val,test인 경우는 torch.no_grad적용
            current_epoch (int): 현재 Epoch
            accumulation_steps (int): gradient accumulation steps

        Returns:
            Tuple[AverageMeter, MetricsMeter]
                - AverageMeter: 평균 Empirical loss
                - MetricsMeter: 평균 Metrics

        Example:
            >>> trainer = BinaryClassifierTrainer(
                    model=model,
                    loss=nn.BCEWithLogitsLoss(),
                    optimizer=optimizer,
                )
            >>> loss, metrics = trainer.run_epoch(dataloader, phase="train", epoch=1)
            >>> mlflow.log_metric("train_loss", loss.avg, step=current_epoch)
            >>> mlflow.log_metrics(metrics.to_dict(), step=current_epoch)
        """
        total_step = len(dataloader)
        if verbose:
            bar = Bar(max=total_step, check_tty=False)

        loss_meter = AverageMeter("loss")
        metrics_meter = MetricsMeter(name=phase)
        for step, batch in enumerate(dataloader, start=1):
            xs, bag_label, patch_labels = batch
            xs = xs.to(self.device)
            bag_label = bag_label.to(self.device)
            patch_labels = patch_labels.to(self.device)

            if phase == "train":
                self.model.train()
                logits, attention_weights = self.model(xs)
            else:
                self.model.eval()
                with torch.no_grad():
                    logits, attention_weights = self.model(xs)

            bag_label = bag_label.view(-1, 1)
            logits = logits.view(-1, 1)
            loss = self.loss(
                bag_label,
                logits,
                patch_labels,
                attention_weights.view(patch_labels.shape),
            )
            loss = loss / accumulation_steps
            if phase == "train":
                loss.backward()
                if step % accumulation_steps == 0 or (step == total_step):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # metric
            loss_meter.update(loss.item() * accumulation_steps, len(bag_label))
            model_confidence: float = torch.sigmoid(logits).item()
            metrics_meter.update(
                [model_confidence],
                [bag_label.item()],
            )

            if verbose:
                bar.suffix = self.make_bar_sentence(
                    phase=phase,
                    epoch=current_epoch,
                    step=step,
                    total_step=total_step,
                    eta=bar.eta,
                    total_loss=loss_meter.avg,
                    metrics_meter=metrics_meter,
                )
                bar.next()

        if verbose:
            bar.finish()

        return (loss_meter, metrics_meter)


class CLAMTrainer(BinaryClassifierTrainer):
    """
    CLAM 모델 전용 trainer.

    bag–level 예측(loss)와 instance–level 평가(loss)를 함께 계산하여
    총 손실을 bag_weight에 따라 가중합한 후 학습하는 구조입니다.

    인자:
      - model: CLAM_SB 또는 CLAM_MB 모델 (forward에서 (logits, Y_prob, Y_hat, A_raw, instance_dict)를 반환)
      - loss_fn: bag–level 손실 함수 (예: nn.CrossEntropyLoss)
      - optimizer: 모델 최적화 옵티마이저
      - bag_weight: bag loss에 대한 가중치 (0 ~ 1 사이 값, instance loss 가중치는 1 - bag_weight)
      - logger: 로깅용 logger (옵션)
    """

    def __init__(
        self,
        model: torch.nn.modules.Module,
        loss: torch.nn.modules.loss._Loss,
        optimizer: torch.optim.Optimizer = None,
        bag_weight: float = 0.5,
        logger: logging.Logger = None,
        device: str = "cuda",
    ):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.bag_weight = bag_weight
        self.logger = logger
        self.device = device

    def run_epoch(
        self,
        dataloader: DataLoader,
        phase: Literal["train", "val", "test"],
        current_epoch: int,
        accumulation_steps: int = 1,
        verbose: bool = True,
    ) -> Tuple[AverageMeter, MetricsMeter]:
        total_step = len(dataloader)
        if verbose:
            bar = Bar(max=total_step, check_tty=False)

        loss_meter = AverageMeter("loss")
        metrics_meter = MetricsMeter(name=phase, accuracy_threshold=0.5)
        for step, batch in enumerate(dataloader, start=1):
            xs, ys = batch
            data = xs.squeeze(0).to(self.device)
            label = ys.long().to(self.device)

            if phase == "train":
                self.model.train()
                # CLAM의 forward는 label과 instance_eval 플래그를 인자로 받음
                logits, Y_prob, Y_hat, A_raw, instance_dict = self.model(
                    data, label=label, instance_eval=True
                )
            else:
                self.model.eval()
                with torch.no_grad():
                    logits, Y_prob, Y_hat, A_raw, instance_dict = self.model(
                        data, label=label, instance_eval=False
                    )

            # bag–level loss
            bag_loss = self.loss(logits, label)
            # instance–level loss
            instance_loss = instance_dict.get("instance_loss", 0.0)

            # total loss
            loss = self.bag_weight * bag_loss + (1 - self.bag_weight) * instance_loss
            loss = loss / accumulation_steps

            if phase == "train":
                loss.backward()
                if step % accumulation_steps == 0 or (step == total_step):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            loss_meter.update(loss.item() * accumulation_steps, len(ys))
            if logits.numel() == 1:
                model_confidence = torch.sigmoid(logits).item()
            else:
                model_confidence = Y_prob[0, 1].item()
            metrics_meter.update(
                [model_confidence],
                [ys.item()],
            )

            if verbose:
                bar.suffix = self.make_bar_sentence(
                    phase=phase,
                    epoch=current_epoch,
                    step=step,
                    total_step=total_step,
                    eta=bar.eta,
                    total_loss=loss_meter.avg,
                    metrics_meter=metrics_meter,
                )
                bar.next()

        if verbose:
            bar.finish()

        return (loss_meter, metrics_meter)
