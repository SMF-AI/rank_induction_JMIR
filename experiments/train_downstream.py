import os
import argparse
import sys
import shlex
import random

import mlflow
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

from rank_induction.datasets import (
    MILDataset,
    MILCachedDataset,
    Batch,
    get_balanced_weight_sequence,
    get_key,
    stratified_subsample_dataset,
)
from rank_induction.networks.mil import AttentionBasedFeatureMIL, DSMIL, CLAM_SB, CLAM_MB
from rank_induction.networks.HIPT import HIPT
from rank_induction.trainer import (
    BinaryClassifierTrainer,
    MixedSupervisionTrainer,
    CLAMTrainer,
)
from rank_induction.losses import AttentionInductionLoss, RankNetLoss, get_pos_weight
from rank_induction.log_ops import TRACKING_URI, get_experiment
from rank_induction.misc import seed_everything, worker_init_fn


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        epilog=(
            "Example usage:\n"
            """python3 experiments/attention_induction/train_downstream.py \\
            --data_dir /vast/AI_team/dataset/CAMELYON16/feature/resnet50_3rd_20x_h5 \\
            --cache_dir /vast/AI_team/dataset/CAMELYON16/instance_labels/attention_induction_20x_224 \\
            --run_name test \\
            --in_features 1024 \\
            --learning attention_induction \\
            --model_type attention_based \\
            --dataset camelyon \\
            --n_classes 1 \\
            --experiment_name attention_induction \\
            --random_state 2025 \\
            --threshold 1 \\
            --max_patiences 7 \\
            --minimal_earlystop_epoch 20 \\
            --num_pos 1024 \\
            --num_neg 1024 \\
            --num_workers 2 \\
            --prefetch_factor 8 \\
            --device 'cuda' \\
            --skip_if_exists
            """
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/vast/AI_team/dataset/CAMELYON16/patch",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/vast/AI_team/dataset/CAMELYON16/cache",
    )
    parser.add_argument(
        "-r",
        "--run_name",
        type=str,
        default="train_from_feature",
    )
    parser.add_argument("--in_features", type=int, default=1024, required=True)

    parser.add_argument(
        "--learning",
        type=str,
        help="training strategy",
        choices=["base", "attention_induction", "ltr"],
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="attention_based",
        choices=["attention_based", "dsmil", "hipt", "clam_sb", "clam_mb"],
        help="Type of MIL model to use",
    )
    parser.add_argument(
        "--inst_loss",
        type=str,
        default="ce",
        choices=["svm", "ce"],
        help="Instance loss for CLAM models (svm or ce)",
    )
    parser.add_argument(
        "--skip_if_exists",
        action="store_true",
        help="Skip training if a finished MLflow run with the same run_name already exists.",
    )
    parser.add_argument("--clam_bag_weight", type=float, default=0.7)
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help=(
            "For development purposes: "
            "runs with only 10 samples per class "
            "to verify code execution without runtime errors"
        ),
    )
    # 데이터셋 종류
    parser.add_argument(
        "--dataset",
        choices=["camelyon", "digest", "smf"],
        default="camelyon",
        help="which WSI collection to use",
    )

    # Optional arguments
    parser.add_argument(
        "--n_classes",
        type=int,
        required=False,
        default=1,
    )
    parser.add_argument("--sampling_ratio", type=float, default=1.0)
    parser.add_argument("--annotation_fraction", type=float, default=1.0)
    parser.add_argument("--ignore_equal", action="store_true")
    parser.add_argument("--margin", default=1.0, type=float, help="Ranknet margin")
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--experiment_name", type=str, default="rank_induction")
    parser.add_argument("-a", "--accumulation_steps", type=int, default=8)
    parser.add_argument("--use_balanced_weight", action="store_true")
    parser.add_argument("--_lambda", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prefetch_factor", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--max_patiences", type=int, default=7)
    parser.add_argument("--minimal_earlystop_epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--random_state", type=int, default=2025)
    parser.add_argument(
        "--num_pos",
        type=int,
        default=1024,
        help="Number of positive instances for RankNet loss calculation",
    )
    parser.add_argument(
        "--num_neg",
        type=int,
        default=1024,
        help="Number of negative instances for RankNet loss calculation",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold value to apply to the attention weight",
    )
    parser.add_argument(
        "--morphology_value",
        type=int,
        default=0,
        help="Morphology value for test polygon extensoin",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def get_model(
    model_type: str,
    in_features: int,
    num_classes: int,
    threshold: float = None,
    return_with: str = "attention_score",
    inst_loss: str = "ce",
    device: str = "cuda",
    **kwargs,
) -> torch.nn.Module:
    """
    모델 타입에 따라 적절한 MIL 모델을 반환하는 factory 함수

    Args:
        model_type: 모델 타입 ("attention_based", "dsmil", "hipt", "clam_sb", "clam_mb")
        in_features: 입력 feature 차원
        num_classes: 클래스 수
        threshold: attention threshold
        return_with: 반환할 attention 타입
        inst_loss: CLAM instance loss ("svm" or "ce")
        device: device string for CLAM SVM loss
        **kwargs: 추가 모델 파라미터

    Returns:
        torch.nn.Module: 선택된 MIL 모델
    """
    if model_type == "attention_based":
        return AttentionBasedFeatureMIL(
            in_features=in_features,
            adaptor_dim=256,
            num_classes=num_classes,
            threshold=threshold,
            return_with=return_with,
        )
    elif model_type == "dsmil":
        return DSMIL(
            in_features=in_features,
            adaptor_dim=256,
            num_classes=num_classes,
            attn_dim=384,
            threshold=threshold,
            return_with=return_with,
        )
    elif model_type == "hipt":
        return HIPT(
            in_dim=in_features,
            grid_hw=(16, 16),
            num_classes=num_classes,
            threshold=threshold,
            return_with=return_with,
        )
    elif model_type in ["clam_sb", "clam_mb"]:
        if inst_loss == "svm":
            from topk.svm import SmoothTop1SVM

            instance_loss_fn = SmoothTop1SVM(n_classes=num_classes).cuda(device=device)
        else:
            instance_loss_fn = nn.CrossEntropyLoss()

        if model_type == "clam_sb":
            return CLAM_SB(in_features, instance_loss_fn=instance_loss_fn)
        else:
            return CLAM_MB(in_features, instance_loss_fn=instance_loss_fn)
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. Must be one of ['attention_based', 'dsmil', 'hipt', 'clam_sb', 'clam_mb']"
        )


def get_train_val_test_dataset_base(
    data_dir,
    random_state=2025,
    sampling_ratio: float = 1.0,
):
    train_batch: Batch = Batch.from_root_path(os.path.join(data_dir, "train"))
    test_batch: Batch = Batch.from_root_path(os.path.join(data_dir, "test"))

    train_dataset = MILDataset(train_batch)

    if sampling_ratio != 1.0:
        train_dataset = stratified_subsample_dataset(
            train_dataset,
            sampling_ratio=sampling_ratio,
        )

    train_dataset, val_dataset = train_dataset.train_test_split(
        test_size=0.2,
        random_state=random_state,
        stratify=True,
    )
    test_dataset = MILDataset(test_batch)

    return train_dataset, val_dataset, test_dataset


def get_train_val_test_dataset_cached(
    data_dir,
    cache_dir,
    random_state=2025,
    sampling_ratio: float = 1.0,
    annotation_fraction: float = 1.0,
):
    """
    Annotation fraction의 경우 train dataset에만 적용하면되서, 마지막에 dataset 만든후

    """
    train_batch: Batch = Batch.from_root_path(os.path.join(data_dir, "train"))
    test_batch: Batch = Batch.from_root_path(os.path.join(data_dir, "test"))

    cache_paths = [
        os.path.join(cache_dir, fn)
        for fn in os.listdir(cache_dir)
        if fn.endswith(".npy")
    ]

    train_keys = set(train_batch.data.keys())
    test_keys = set(test_batch.data.keys())
    train_cache_paths = [p for p in cache_paths if get_key(p) in train_keys]
    test_cache_paths = [p for p in cache_paths if get_key(p) in test_keys]

    train_batch.add_instance_labels_from_cache(train_cache_paths)
    test_batch.add_instance_labels_from_cache(test_cache_paths)

    train_dataset = MILCachedDataset(train_batch)

    if sampling_ratio != 1.0:
        train_dataset = stratified_subsample_dataset(
            train_dataset,
            sampling_ratio=sampling_ratio,
        )

    train_dataset, val_dataset = train_dataset.train_test_split(
        test_size=0.2,
        random_state=random_state,
        stratify=True,
    )

    if annotation_fraction != 1.0:
        train_dataset.batch.keep_annotation(fraction=annotation_fraction)

    test_dataset = MILCachedDataset(test_batch)

    return train_dataset, val_dataset, test_dataset


if __name__ == "__main__":
    # mp.set_start_method("spawn")

    args = get_args()
    seed_everything(args.random_state)

    # If requested, skip training when a finished MLflow run with the same run_name exists
    if args.skip_if_exists:
        mlflow.set_tracking_uri(TRACKING_URI)
        exp = get_experiment(args.experiment_name)
        try:
            from mlflow.tracking import MlflowClient

            client = MlflowClient()
            filter_string = f"params.run_name = '{args.run_name}' and attributes.status = 'FINISHED'"
            existing = client.search_runs(
                [exp.experiment_id], filter_string=filter_string, max_results=1
            )
            if len(existing) > 0:
                print(
                    f"[SKIP] Finished MLflow run already exists for run_name={args.run_name}. Skipping training."
                )
                raise SystemExit(0)
        except Exception as e:
            print(f"[WARN] Failed to query MLflow for existing runs: {e}")

    if args.learning == "base":
        train_dataset, val_dataset, test_dataset = get_train_val_test_dataset_base(
            args.data_dir,
            random_state=args.random_state,
            sampling_ratio=args.sampling_ratio,
        )
    else:
        train_dataset, val_dataset, test_dataset = get_train_val_test_dataset_cached(
            args.data_dir,
            args.cache_dir,
            random_state=args.random_state,
            sampling_ratio=args.sampling_ratio,
            annotation_fraction=args.annotation_fraction,
        )

    weights = get_balanced_weight_sequence(train_dataset)
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        worker_init_fn=worker_init_fn,
    )
    val_dataloader = DataLoader(
        val_dataset,
        num_workers=args.num_workers,
        shuffle=True,
        prefetch_factor=args.prefetch_factor,
    )
    test_dataloader = DataLoader(
        test_dataset, num_workers=args.num_workers, shuffle=True
    )

    if args.learning == "attention_induction":
        return_with = "attention_weight"
    elif args.learning == "ltr":
        return_with = "attention_score"
    elif args.learning == "base":
        return_with = "contribution"
    else:
        raise ValueError(
            f"learning({args.learning}) must be one of 'attention_induction' or 'ltr'"
        )

    model = get_model(
        model_type=args.model_type,
        in_features=args.in_features,
        num_classes=args.n_classes,
        threshold=args.threshold,
        return_with=return_with,
        inst_loss=args.inst_loss,
        device=args.device,
    ).to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

    if args.model_type in ["clam_sb", "clam_mb"]:
        trainer = CLAMTrainer(
            model=model,
            loss=torch.nn.CrossEntropyLoss(),
            optimizer=optimizer,
            bag_weight=args.clam_bag_weight,
        )
    else:
        weight = None
        if args.use_balanced_weight:
            weight = get_pos_weight(train_dataset.bag_labels).to(args.device)

        if args.learning == "base":
            trainer = BinaryClassifierTrainer(
                model=model,
                loss=torch.nn.BCEWithLogitsLoss(pos_weight=weight),
                optimizer=optimizer,
            )
        elif args.learning == "attention_induction":
            trainer = MixedSupervisionTrainer(
                model=model,
                loss=AttentionInductionLoss(_lambda=args._lambda, pos_weight=weight),
                optimizer=optimizer,
            )
        elif args.learning == "ltr":
            trainer = MixedSupervisionTrainer(
                model=model,
                loss=RankNetLoss(
                    _lambda=args._lambda,
                    sigma=args.sigma,
                    ignore_equal=args.ignore_equal,
                    num_pos=args.num_pos,
                    num_neg=args.num_neg,
                    device=args.device,
                ),
                optimizer=optimizer,
            )

    mlflow.set_tracking_uri(TRACKING_URI)
    exp = get_experiment(args.experiment_name)
    with mlflow.start_run(experiment_id=exp.experiment_id, run_name=args.run_name):
        # 현재 실행 커맨드를 그대로 태그로 저장
        command_tag = " ".join(
            shlex.quote(x) for x in ([os.path.basename(sys.executable)] + sys.argv)
        )
        mlflow.set_tag("command", command_tag)
        mlflow.log_params(vars(args))
        mlflow.log_artifact(os.path.abspath(__file__))

        trainer.train(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            max_patiences=args.max_patiences,
            minimal_earlystop_epoch=args.minimal_earlystop_epoch,
            n_epochs=args.epochs,
            accumulation_steps=args.accumulation_steps,
            use_mlflow=True,
            verbose=args.verbose,
        )
        trainer.test(test_dataloader, use_mlflow=True, verbose=args.verbose)
        mlflow.pytorch.log_model(model, artifact_path="model")
