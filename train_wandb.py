import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
import argparse
import warnings
from data_module import histo_DataModule
from transabmil import TransABMIL
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
from utils import str2bool
from survivmil_CI import SURVIVMIL # novel concordance-based-loss


CUDA_LAUNCH_BLOCKING=1

warnings.filterwarnings("ignore")


def get_args():
    parser = argparse.ArgumentParser(description="Training survivMIL for the prediction of outcome")

    parser.add_argument(
        "--csv_dir",
        type=str,
        default=None,
        help="Path to CSV file containing data paths",
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default="",
        help="Path to directory containing images",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for training"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=100, help="Maximum number of training epochs"
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=os.cpu_count(),
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="CV_logs",
        help="Directory to save the model weights to",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for training",
    )
    parser.add_argument(
        "--lr", type=float, default=0.0002, help="Learning rate for optimizer"
    )

    parser.add_argument(
        "--lossweight", 
        type=float, 
        default=0.5, 
        help="Weight of the negative log likelihood loss"
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
        help="Weight decay parameter for optimizer",
    )
    parser.add_argument(
        "--input_size",
        type=int,
        default=1024, 
        help="Hidden dimension of projection head",
    )
    parser.add_argument(
        "--log_dir", type=str, default="CV_logs/", help="directory to save logs"
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=None,
        help="Path to pretrained model",
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="",
        help="Name of the project",
    )
    parser.add_argument(
        "--logger",
        type=str,
        default="wandb",
        choices=["wandb", "tensorboard", "neptune"],
        help="Whether to use wandb for logging",
    )
    parser.add_argument(
        "--fold",
        type=int,
        default=1,
        help="Fold to train on",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["TransABMIL", "SURVIVMIL", "DSMIL" , "mean_pool", "max_pool", "lse_pool", 'ABMIL','TransMIL', 'AMIL', 'MCAT'],
        default="DSMIL",
        help="Choice of model.",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=['ci_loss', 'nll_loss'],
        default="nll_loss",
        help="Choice of loss function.",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="train",
        choices=["train", "test", "calculate_qms", "calculate_attention"],
        help="Stage of training",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=2,
        help="Number of classes",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="PredictOutcome",
        choices=[
            "PredictOutcome",
        ],
        help="Task to perform",
    )
    parser.add_argument(
        "--linear_iclass",
        type=str2bool,
        default=False,  # Having this True may cause issues with attention
        help="Whether to use linear classifier for i_class",
    )
    parser.add_argument(
        "--concat",
        type=str2bool,
        default=False,  
        help="Whether to concatenate the modalities",
    )
    parser.add_argument(
        "--normalise_classes",
        type=str2bool,
        default=False,
        help="Whether to normalise classes between -1 and 1 for the "
        "instance classifier when calculating attention",
    )
    parser.add_argument(
        "--augment_type",
        type=str,
        default=None,
        help="Type of augmentation to perform",
    )
    parser.add_argument(
        "--sub_aug_type",
        type=str,
        default=None,
        help="Type of sub augmentation to perform",
    )
    parser.add_argument(
        "--one_vs_target",
        type=str,
        default='high',
        help="Target for one vs the rest training",
    )
    parser.add_argument(
        '--wandb_run_name',
        type=str,
        default = None, 
        help='the name of your wandb run',
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=100,
        help="Patience for early stopping",
    )
    parser.add_argument(
        "--multi_class",
        type=str2bool,
        default=False,
        help="Whether to use multi class loss",
    )
    parser.add_argument(
        "--opt",
        type=str, 
        default='lookahead_radam',
    )
    parser.add_argument(
        "--multimodal",
        type=str,
        default="all",
        choices=["all", "wsi", "ehr"],
        help="Modalities used for training",
    )

    return parser.parse_args()


def build_model(args):
    if args.model_type == "SURVIVMIL":
        i_class = "i_class"
    if args.model_type == "MCAT":
        i_class = "i_class"
    if args.model_type == "DSMIL":
        i_class = "i_class"
    elif args.model_type == "TransABMIL":
        i_class = "trans"
    elif args.model_type == "ABMIL":
        i_class = "i_class"
    else:
        i_class = "none"

    setattr(
        args,
        "log_dir",
        os.path.join(
            args.log_dir, args.project_name
        ),
    )


    if args.num_classes == 2:
        criterion = (nn.BCEWithLogitsLoss())
    else:
        criterion = (nn.CrossEntropyLoss())

    if args.num_classes == 2:
        output_class = 1
    else:
        output_class = args.num_classes


    if args.model_type == 'TransABMIL':
        model = TransABMIL(
            num_classes=args.num_classes,
            criterion=criterion,
            model_type=args.model_type,
            i_class=i_class,
            log_dir=args.log_dir,
            output_class=output_class,
            lr=args.lr,
        )

    elif args.model_type == 'MCAT':
        model = mcat(
            loss_type = args.loss, 
            num_classes=args.num_classes,
            criterion=criterion,
            model_type=args.model_type,
            log_dir=args.log_dir,
            output_class=output_class,
        )

    elif args.model_type == 'SURVIVMIL':
        model = SURVIVMIL(
            multimodal = args.multimodal,
            lossweight = args.lossweight, 
            num_classes=args.num_classes,
            criterion=criterion,
            model_type=args.model_type,
            log_dir=args.log_dir,
            output_class=output_class,
        )

    elif args.model_type == 'DSMIL':
        model = DSMIL(
            num_classes=args.num_classes,
            multimodal = args.multimodal,
            lossweight = args.lossweight, 
            loss_type = args.loss,   
            criterion=criterion,
            model_type=args.model_type,
            log_dir=args.log_dir,
            output_class=output_class,
        )


    elif args.model_type == 'AMIL':
        model = AMIL(
            n_classes= args.num_classes,
            num_classes=args.num_classes,
            multimodal = args.multimodal,
            lossweight = args.lossweight, 
            loss_type = args.loss,   
            criterion=criterion,
            model_type=args.model_type,
            log_dir=args.log_dir,
            output_class=output_class,
        )

    elif args.model_type == 'TransMIL':
        model = transmil(
            num_classes=args.num_classes,
            n_classes= args.num_classes,
            multimodal = args.multimodal,
            lossweight = args.lossweight, 
            loss_type = args.loss,   
            criterion=criterion,
            model_type=args.model_type,
            log_dir=args.log_dir,
            output_class=output_class,
        )

    return model

# speeding up training 
torch.set_float32_matmul_precision('high')


def train(args):
    
    print(f"Training for the prediction of patient outcome")
    print('Modalities: ', args.multimodal)


    # Setting the seed
    pl.seed_everything(42)
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.00,
        patience=args.patience,
        verbose=True,
        mode="min",
    )

    print('Fold references taken from:', args.csv_dir)

    # checkpoint_callback = ModelCheckpoint(monitor="val_loss")

    checkpoint_callback = ModelCheckpoint(
    monitor="val_c_index",
    mode="max",  # 'max' to save the model with the highest c-index
    save_top_k=1,  # Save only the best model
    verbose=True,
    save_last=True  # Optionally, save the last checkpoint
    )
    
    lr_monitor = LearningRateMonitor(logging_interval="step")

    smpeds_data = histo_DataModule(
        csv_path=args.csv_dir,
        h5_path=args.img_dir,
        batch_size=args.batch_size,
        task=args.task,
        augment_type=args.augment_type,
        sub_aug_type=args.sub_aug_type,
        concat = args.concat,
        args=args,
    )

    smpeds_data.setup()
    model = build_model(args)

    if args.logger == "wandb":
        logger = WandbLogger(
            project=args.project_name,
            name = args.wandb_run_name,
            log_model=True,
            save_dir=args.log_dir + '/' + args.wandb_run_name,
        )

    elif args.logger == "tensorboard":
        logger = pl.loggers.TensorBoardLogger(
            save_dir=args.log_dir ,
        )

    else:
        raise ValueError(f"Invalid logger {args.logger}")

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.num_gpus,
        max_epochs=args.max_epochs,
        callbacks=[early_stop_callback, checkpoint_callback, lr_monitor],
        default_root_dir=args.log_dir,
        logger=logger,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, smpeds_data)
    print(f"Finished training for the prediction of patient outcome")
    test_results = trainer.test(model=model, datamodule=smpeds_data)

    print(test_results)
    test_results_df = pd.DataFrame(test_results)
    test_results_df.to_csv(args.log_dir + '/' + args.wandb_run_name + '/' + 'result.csv', index=False)
    print('Test results saved to path: {}'.format(args.log_dir + '/' + args.wandb_run_name ))


def test(args):
    pass





if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")

    args = get_args()
    if args.stage == "train":
        print("Starting training")
        train(args)
    else:
        raise ValueError(f"Invalid stage {args.stage}")