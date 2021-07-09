import math
import argparse 

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import ModelPruning
from pytorch_lightning.utilities.cloud_io import load as pl_load

from ray import tune
from ray.tune import CLIReporter, JupyterNotebookReporter
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback

from models import ExperimentModel
from datamodules import cifar_datamodule

parser = argparse.ArgumentParser(description="Dropback & prune experiments.")
parser.add_argument("--experiment_name", default="baseline")

def main():
    args = parser.parse_args()

    checkpoint_callback = ModelCheckpoint(
        monitor='ptl/val_loss',
        filename='epoch{epoch:02d}-val_accuracy{ptl/val_accuracy_top1:.2f}',
        save_top_k=3,
        mode='min',
        auto_insert_metric_name=False
        )

    prune_callback = ModelPruning(
        pruning_fn='l1_unstructured',
        parameter_names=["weight", "bias"],
        amount = lambda epoch: 0.1 if (epoch > 299 and epoch % 100 == 0) else 0,
        use_global_unstructured=True,
        verbose=1,
        )
    
    callback = [
        checkpoint_callback,
        # prune_callback,
        TuneReportCallback(
            metrics = {
                "loss": "ptl/val_loss",
                "mean_accuracy": "ptl/val_accuracy_top1",
                "current_lr": "current_lr",
            },
            on="validation_end")
    ]

    cifar10_dm = cifar_datamodule()
    # cifar100_dm = cifar_datamodule(dataset="cifar100")

    checkpoint_path = None

    if args.experiment_name == "baseline":
        callback.append(checkpoint_callback)
    elif args.experiment_name == "prune":
        callback.append(prune_callback)
    elif args.experiment_name == "dropback":
        callback.append(checkpoint_callback)
    

    tune_asha(
        num_samples=1, 
        num_epochs=400, 
        gpus_per_trial=1,
        deterministic=False,
        datamodule=cifar10_dm,
        # num_classes=100,
        callback=callback,
        checkpoint_path=checkpoint_path,
        experiment_name=args.experiment_name
        )


def training(
    config, 
    num_epochs=10, 
    num_gpus=0,
    deterministic=False,
    datamodule=None,
    num_classes=100,
    callback: list = None,
    checkpoint_path=None,
    experiment_name = "baseline"
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
        model = ExperimentModel.load_from_checkpoint(
            checkpoint_path, 
            config=config, 
            num_classes=num_classes,
            experiment=experiment_name
            )  
    else:
        model = ExperimentModel(
            config=config,
            experiment=experiment_name
            )

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
    experiment_name="baseline"
    ):

    config = {
#         "lr": tune.loguniform(1e-4, 1e-1),
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 4e-5,
    }

#     config = {
# #         "lr": tune.loguniform(1e-4, 1e-1),
#         "lr": 0.1,
#         "momentum": 0.9,
#         "weight_decay": 4e-5,
#         # "batch_size": tune.choice([32, 64, 128]),
#         "track_size": 140000,
#         "init_decay": 0.1,
#     }

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
            checkpoint_path=checkpoint_path,
            experiment_name=experiment_name
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
        name=experiment_name)

    print("Best hyperparameters found were: ", analysis.best_config)

if __name__ == '__main__':
    main()