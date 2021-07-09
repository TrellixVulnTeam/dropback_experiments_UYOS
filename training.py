import math
import argparse 

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.cloud_io import load as pl_load

from ray import tune
from ray.tune import CLIReporter, JupyterNotebookReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback

from models import ExperimentModel
from datamodules import cifar_datamodule

def training(
    config, 
    num_epochs=10, 
    num_gpus=0,
    deterministic=False,
    datamodule=None,
    num_classes=100,
    callback: list = None,
    checkpoint_path=None
    ):

    if deterministic:
        seed_everything(42, workers=True)
    
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        # If fractional GPUs passed in, convert to int.
        gpus=math.ceil(num_gpus),
        logger=TensorBoardLogger(
            save_dir=tune.get_trial_dir(), name="", version="."),
        progress_bar_refresh_rate=0,
        deterministic=deterministic,
        callbacks=callback,
    )

    if checkpoint_path:
        model = ExperimentModel.load_from_checkpoint(checkpoint_path, config=config, num_classes=num_classes)  
    else:
        model = ExperimentModel(config=config)

    trainer.fit(
        model, 
        datamodule=datamodule,
    ) 

def tune_asha(
    num_samples=10, 
    num_epochs=10, 
    gpus_per_trial=0,
    deterministic=False,
    datamodule=None,
    num_classes=100,
    callback: list = None,
    checkpoint_path=None,
    experiement_name="baseline"
    ):

    config = {
#         "lr": tune.loguniform(1e-4, 1e-1),
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 4e-5,
    }

    scheduler = ASHAScheduler(
        max_t=num_epochs,
        grace_period=20,
        reduction_factor=2)

    # reporter = JupyterNotebookReporter(
    #     overwrite=False,
    #     parameter_columns=["lr", "momentum"],
    #     metric_columns=["loss", "mean_accuracy", "training_iteration", "current_lr"]
    # )
    reporter = CLIReporter(
        parameter_columns=["lr", "momentum"],
        metric_columns=["loss", "mean_accuracy", "training_iteration", "current_lr"])

    analysis = tune.run(
        tune.with_parameters(
            training,
            num_epochs=num_epochs,
            num_gpus=gpus_per_trial,
            deterministic=deterministic,
            datamodule=datamodule,
            num_classes=num_classes,
            callback=callback,
            checkpoint_path=checkpoint_path
        ),
        resources_per_trial={
            "cpu": 2,
            "gpu": gpus_per_trial
        },
        metric="loss",
        mode="min",
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        name=experiement_name)

    print("Best hyperparameters found were: ", analysis.best_config)

        
if __name__ == '__main__':
    callback = [
        ModelCheckpoint(
            monitor='ptl/val_loss',
            filename='epoch{epoch:02d}-val_accuracy{ptl/val_accuracy_top1:.2f}',
            save_top_k=3,
            mode='min',
            auto_insert_metric_name=False
        ),
        TuneReportCallback(
            metrics = {
                "loss": "ptl/val_loss",
                "mean_accuracy": "ptl/val_accuracy_top1",
                "current_lr": "current_lr",
            },
            on="validation_end")
    ]

    cifar10_dm = cifar_datamodule()
    tune_asha(
    num_samples=1, 
    num_epochs=50, 
    gpus_per_trial=1,
    deterministic=True,
    datamodule=cifar10_dm,
    num_classes=10,
    callback=callback,
    checkpoint_path=None
    )


# def training_w_checkpoint(config, checkpoint_dir = None, num_epochs=10, num_gpus=0):
    
#     seed_everything(42, workers=True)

#     trainer = pl.Trainer(
#         max_epochs=num_epochs,
#         # If fractional GPUs passed in, convert to int.
#         gpus=math.ceil(num_gpus),
#         logger=TensorBoardLogger(
#             save_dir=tune.get_trial_dir(), name="", version="."),
#         progress_bar_refresh_rate=0,
#         deterministic=True,
#         callbacks=
#         [
#             ModelCheckpoint(
#                 monitor='ptl/val_loss',
#                 filename='epoch{epoch:02d}-top1_accuracy{ptl/val_accuracy_top1:.2f}',
#                 save_top_k=3,
#                 mode='min',
#             ),
#             TuneReportCheckpointCallback(
#                 metrics = {
#                     "loss": "ptl/val_loss",
#                     "mean_accuracy": "ptl/val_accuracy_top1",
#                     "current_lr": "current_lr",
#                 },
#                 filename="checkpoint",
#                 on="validation_end")
#         ]
#     )
    
#     if checkpoint_dir:
#         # Currently, this leads to errors:
#         # model = LightningMNISTClassifier.load_from_checkpoint(
#         #     os.path.join(checkpoint, "checkpoint"))
#         # Workaround:
#         ckpt = pl_load(
#             os.path.join(checkpoint_dir, "checkpoint"),
#             map_location=lambda storage, loc: storage)
#         model = ExperimentModel._load_model_state(
#             ckpt, config=config)
#         trainer.current_epoch = ckpt["epoch"]
#     else:
#         model = ExperimentModel(config=config)
        
#     trainer.fit(model, datamodule=cifar10_dm) 


# def tune_pbt(
#     num_samples=10, 
#     num_epochs=10, 
#     gpus_per_trial=0, 
#     ):
#     config = {
#         "lr": 0.1,
#         "momentum": 0.9,
#         "weight_decay": 4e-5,
#     }
    
#     scheduler = PopulationBasedTraining(
#         perturbation_interval=4,
#         hyperparam_mutations={
#             "lr": tune.loguniform(1e-4, 1e-1),
#         }
#     )

#     reporter = JupyterNotebookReporter(
#         overwrite=False,
#         parameter_columns=["lr", "momentum"],
#         metric_columns=["loss", "mean_accuracy", "training_iteration", "current_lr"]
#     )
# #     reporter = CLIReporter(
# #         parameter_columns=["lr", "momentum"],
# #         metric_columns=["loss", "mean_accuracy", "training_iteration", "current_lr"])

#     analysis = tune.run(
#         tune.with_parameters(
#             training_w_checkpoint,
#             num_epochs=num_epochs,
#             num_gpus=gpus_per_trial,
#         ),
#         resources_per_trial={
#             "cpu": 2,
#             "gpu": gpus_per_trial
#         },
#         metric="loss",
#         mode="min",
#         config=config,
#         num_samples=num_samples,
#         scheduler=scheduler,
#         progress_reporter=reporter,
#         name="baseline_pbt")

#     print("Best hyperparameters found were: ", analysis.best_config)