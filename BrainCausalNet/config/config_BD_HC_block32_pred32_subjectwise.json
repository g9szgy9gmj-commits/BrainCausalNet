{
    "name": "BD_HC Subjectwise Block32 Predict32 Causality Learning",
    "n_gpu": 1,
    "arch": {
        "type": "PredictModel",
        "args": {
            "d_model": 256,
            "n_head": 4,
            "n_layers": 1,
            "ffn_hidden": 512,
            "drop_prob": 0,
            "tau": 100
        }
    },
    "data_loader": {
        "type": "SubjectBlockFutureNpyTimeseriesDataLoader",
        "args": {
            "data_dir": "data/BD_HC/BD_HC_sig.npy",
            "subject_idx": 0,
            "batch_size": 256,
            "time_step": 32,
            "output_window": 32,
            "feature_dim": 1,
            "output_dim": 1,
            "shuffle": true,
            "validation_split": 0.1,
            "num_workers": 2
        }
    },
    "optimizer": {
        "type": "Adam",
        "args": {
            "lr": 0.01,
            "weight_decay": 0,
            "amsgrad": true
        }
    },
    "loss": "masked_mse_torch",
    "metrics": ["masked_mse_torch"],
    "lr_scheduler": {
        "type": "StepLR",
        "args": {
            "step_size": 30,
            "gamma": 0.1
        }
    },
    "trainer": {
        "epochs": 30,
        "save_dir": "saved/",
        "save_period": 1,
        "verbosity": 0,
        "monitor": "min val_loss",
        "early_stop": 10,
        "lam": 0,
        "tensorboard": true
    }
}
