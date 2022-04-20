{
    "SVM": {
        "sklearn_svc_0": {
            "kernel": "linear",
            "gamma": "auto",
            "C": 1.0,
            "degree": 0,
            "cache_size": 500,
            "probability": true
        },
        "sklearn_svc_1": {
            "kernel": "linear",
            "gamma": "scale",
            "C": 1.0,
            "degree": 0,
            "cache_size": 500,
            "probability": true
        }
    },
    "GBM": {
        "xgboost_gbm_0": {
            "boosters": "gbtree",
            "objectives": "binary:logistic",
            "num_class": 2,
            "eval_metric": "mlogloss",
            "etas": 0.3,
            "gammas": 0.1,
            "max_depths": 6,
            "min_child_weights": 2,
            "alphas": 0
        },
        "xgboost_gbm_1": {
            "boosters": "gbtree",
            "objectives": "binary:logistic",
            "num_class": 2,
            "eval_metric": "mlogloss",
            "etas": 0.3,
            "gammas": 0.1,
            "max_depths": 6,
            "min_child_weights": 1,
            "alphas": 0
        },
        "xgboost_gbm_2": {
            "boosters": "gbtree",
            "objectives": "binary:logistic",
            "num_class": 2,
            "eval_metric": "mlogloss",
            "etas": 0.3,
            "gammas": 0,
            "max_depths": 6,
            "min_child_weights": 2,
            "alphas": 0
        },
        "xgboost_gbm_3": {
            "boosters": "gbtree",
            "objectives": "binary:logistic",
            "num_class": 2,
            "eval_metric": "mlogloss",
            "etas": 0.3,
            "gammas": 0,
            "max_depths": 6,
            "min_child_weights": 1,
            "alphas": 0
        },
        "xgboost_gbm_4": {
            "boosters": "gbtree",
            "objectives": "binary:logistic",
            "num_class": 2,
            "eval_metric": "mlogloss",
            "etas": 0.1,
            "gammas": 0.1,
            "max_depths": 6,
            "min_child_weights": 2,
            "alphas": 0
        },
        "xgboost_gbm_5": {
            "boosters": "gbtree",
            "objectives": "binary:logistic",
            "num_class": 2,
            "eval_metric": "mlogloss",
            "etas": 0.1,
            "gammas": 0.1,
            "max_depths": 6,
            "min_child_weights": 1,
            "alphas": 0
        },
        "xgboost_gbm_6": {
            "boosters": "gbtree",
            "objectives": "binary:logistic",
            "num_class": 2,
            "eval_metric": "mlogloss",
            "etas": 0.1,
            "gammas": 0,
            "max_depths": 6,
            "min_child_weights": 2,
            "alphas": 0
        },
        "xgboost_gbm_7": {
            "boosters": "gbtree",
            "objectives": "binary:logistic",
            "num_class": 2,
            "eval_metric": "mlogloss",
            "etas": 0.1,
            "gammas": 0,
            "max_depths": 6,
            "min_child_weights": 1,
            "alphas": 0
        },
        "xgboost_gbm_8": {
            "boosters": "gbtree",
            "objectives": "multi:softmax",
            "num_class": 2,
            "eval_metric": "mlogloss",
            "etas": 0.3,
            "gammas": 0.1,
            "max_depths": 6,
            "min_child_weights": 2,
            "alphas": 0
        },
        "xgboost_gbm_9": {
            "boosters": "gbtree",
            "objectives": "multi:softmax",
            "num_class": 2,
            "eval_metric": "mlogloss",
            "etas": 0.3,
            "gammas": 0.1,
            "max_depths": 6,
            "min_child_weights": 1,
            "alphas": 0
        },
        "xgboost_gbm_10": {
            "boosters": "gbtree",
            "objectives": "multi:softmax",
            "num_class": 2,
            "eval_metric": "mlogloss",
            "etas": 0.3,
            "gammas": 0,
            "max_depths": 6,
            "min_child_weights": 2,
            "alphas": 0
        },
        "xgboost_gbm_11": {
            "boosters": "gbtree",
            "objectives": "multi:softmax",
            "num_class": 2,
            "eval_metric": "mlogloss",
            "etas": 0.3,
            "gammas": 0,
            "max_depths": 6,
            "min_child_weights": 1,
            "alphas": 0
        },
        "xgboost_gbm_12": {
            "boosters": "gbtree",
            "objectives": "multi:softmax",
            "num_class": 2,
            "eval_metric": "mlogloss",
            "etas": 0.1,
            "gammas": 0.1,
            "max_depths": 6,
            "min_child_weights": 2,
            "alphas": 0
        },
        "xgboost_gbm_13": {
            "boosters": "gbtree",
            "objectives": "multi:softmax",
            "num_class": 2,
            "eval_metric": "mlogloss",
            "etas": 0.1,
            "gammas": 0.1,
            "max_depths": 6,
            "min_child_weights": 1,
            "alphas": 0
        },
        "xgboost_gbm_14": {
            "boosters": "gbtree",
            "objectives": "multi:softmax",
            "num_class": 2,
            "eval_metric": "mlogloss",
            "etas": 0.1,
            "gammas": 0,
            "max_depths": 6,
            "min_child_weights": 2,
            "alphas": 0
        },
        "xgboost_gbm_15": {
            "boosters": "gbtree",
            "objectives": "multi:softmax",
            "num_class": 2,
            "eval_metric": "mlogloss",
            "etas": 0.1,
            "gammas": 0,
            "max_depths": 6,
            "min_child_weights": 1,
            "alphas": 0
        }
    },
    "CNN": {
        "Epoch": 2,
        "Batch_Size": [
            5,
            10
        ],
        "1D": {
            "pytorch_cnn_1d_0": {
                "conv_kernel_size": 32,
                "conv_kernel_num": 64,
                "maxpool_kernel_size": 3,
                "densed_size": 128,
                "num_layers": 4,
                "learning_rate": 0.01
            },
            "pytorch_cnn_1d_1": {
                "conv_kernel_size": 32,
                "conv_kernel_num": 64,
                "maxpool_kernel_size": 3,
                "densed_size": 128,
                "num_layers": 1,
                "learning_rate": 0.01
            },
            "pytorch_cnn_1d_2": {
                "conv_kernel_size": 32,
                "conv_kernel_num": 64,
                "maxpool_kernel_size": 3,
                "densed_size": 64,
                "num_layers": 2,
                "learning_rate": 0.01
            },
            "pytorch_cnn_1d_3": {
                "conv_kernel_size": 32,
                "conv_kernel_num": 64,
                "maxpool_kernel_size": 3,
                "densed_size": 64,
                "num_layers": 1,
                "learning_rate": 0.01
            },
            "pytorch_cnn_1d_4": {
                "conv_kernel_size": 32,
                "conv_kernel_num": 64,
                "maxpool_kernel_size": 2,
                "densed_size": 128,
                "num_layers": 2,
                "learning_rate": 0.01
            },
            "pytorch_cnn_1d_5": {
                "conv_kernel_size": 32,
                "conv_kernel_num": 64,
                "maxpool_kernel_size": 2,
                "densed_size": 128,
                "num_layers": 1,
                "learning_rate": 0.01
            },
            "pytorch_cnn_1d_6": {
                "conv_kernel_size": 32,
                "conv_kernel_num": 64,
                "maxpool_kernel_size": 2,
                "densed_size": 64,
                "num_layers": 2,
                "learning_rate": 0.01
            },
            "pytorch_cnn_1d_7": {
                "conv_kernel_size": 32,
                "conv_kernel_num": 64,
                "maxpool_kernel_size": 2,
                "densed_size": 64,
                "num_layers": 1,
                "learning_rate": 0.01
            },
            "pytorch_cnn_1d_8": {
                "conv_kernel_size": 32,
                "conv_kernel_num": 32,
                "maxpool_kernel_size": 3,
                "densed_size": 128,
                "num_layers": 2,
                "learning_rate": 0.01
            },
            "pytorch_cnn_1d_9": {
                "conv_kernel_size": 32,
                "conv_kernel_num": 32,
                "maxpool_kernel_size": 3,
                "densed_size": 128,
                "num_layers": 1,
                "learning_rate": 0.01
            },
            "pytorch_cnn_1d_10": {
                "conv_kernel_size": 32,
                "conv_kernel_num": 32,
                "maxpool_kernel_size": 3,
                "densed_size": 64,
                "num_layers": 2,
                "learning_rate": 0.01
            },
            "pytorch_cnn_1d_11": {
                "conv_kernel_size": 32,
                "conv_kernel_num": 32,
                "maxpool_kernel_size": 3,
                "densed_size": 64,
                "num_layers": 1,
                "learning_rate": 0.01
            },
            "pytorch_cnn_1d_12": {
                "conv_kernel_size": 32,
                "conv_kernel_num": 32,
                "maxpool_kernel_size": 2,
                "densed_size": 128,
                "num_layers": 2,
                "learning_rate": 0.01
            },
            "pytorch_cnn_1d_13": {
                "conv_kernel_size": 32,
                "conv_kernel_num": 32,
                "maxpool_kernel_size": 2,
                "densed_size": 128,
                "num_layers": 1,
                "learning_rate": 0.01
            },
            "pytorch_cnn_1d_14": {
                "conv_kernel_size": 32,
                "conv_kernel_num": 32,
                "maxpool_kernel_size": 2,
                "densed_size": 64,
                "num_layers": 2,
                "learning_rate": 0.01
            },
            "pytorch_cnn_1d_15": {
                "conv_kernel_size": 32,
                "conv_kernel_num": 32,
                "maxpool_kernel_size": 2,
                "densed_size": 64,
                "num_layers": 1,
                "learning_rate": 0.01
            }
        },
        "Hybrid": {
            "pytorch_cnn_hybrid_0": {
                "matrix_size": [
                    292,
                    292
                ],
                "conv_kernel_num": 64,
                "maxpool_kernel_size": 3,
                "densed_size": 128,
                "num_layers": 2,
                "learning_rate": 0.01
            },
            "pytorch_cnn_hybrid_1": {
                "matrix_size": [
                    292,
                    292
                ],
                "conv_kernel_num": 64,
                "maxpool_kernel_size": 3,
                "densed_size": 128,
                "num_layers": 1,
                "learning_rate": 0.01
            },
            "pytorch_cnn_hybrid_2": {
                "matrix_size": [
                    292,
                    292
                ],
                "conv_kernel_num": 64,
                "maxpool_kernel_size": 3,
                "densed_size": 64,
                "num_layers": 2,
                "learning_rate": 0.01
            },
            "pytorch_cnn_hybrid_3": {
                "matrix_size": [
                    292,
                    292
                ],
                "conv_kernel_num": 64,
                "maxpool_kernel_size": 3,
                "densed_size": 64,
                "num_layers": 2,
                "learning_rate": 0.01
            },
            "pytorch_cnn_hybrid_4": {
                "matrix_size": [
                    292,
                    292
                ],
                "conv_kernel_num": 64,
                "maxpool_kernel_size": 2,
                "densed_size": 128,
                "num_layers": 2,
                "learning_rate": 0.01
            },
            "pytorch_cnn_hybrid_5": {
                "matrix_size": [
                    292,
                    292
                ],
                "conv_kernel_num": 64,
                "maxpool_kernel_size": 2,
                "densed_size": 128,
                "num_layers": 1,
                "learning_rate": 0.01
            },
            "pytorch_cnn_hybrid_6": {
                "matrix_size": [
                    292,
                    292
                ],
                "conv_kernel_num": 64,
                "maxpool_kernel_size": 2,
                "densed_size": 64,
                "num_layers": 2,
                "learning_rate": 0.01
            },
            "pytorch_cnn_hybrid_7": {
                "matrix_size": [
                    292,
                    292
                ],
                "conv_kernel_num": 64,
                "maxpool_kernel_size": 2,
                "densed_size": 64,
                "num_layers": 1,
                "learning_rate": 0.01
            },
            "pytorch_cnn_hybrid_8": {
                "matrix_size": [
                    292,
                    292
                ],
                "conv_kernel_num": 32,
                "maxpool_kernel_size": 3,
                "densed_size": 128,
                "num_layers": 2,
                "learning_rate": 0.01
            },
            "pytorch_cnn_hybrid_9": {
                "matrix_size": [
                    292,
                    292
                ],
                "conv_kernel_num": 32,
                "maxpool_kernel_size": 3,
                "densed_size": 128,
                "num_layers": 1,
                "learning_rate": 0.01
            },
            "pytorch_cnn_hybrid_10": {
                "matrix_size": [
                    292,
                    292
                ],
                "conv_kernel_num": 32,
                "maxpool_kernel_size": 3,
                "densed_size": 64,
                "num_layers": 2,
                "learning_rate": 0.01
            },
            "pytorch_cnn_hybrid_11": {
                "matrix_size": [
                    292,
                    292
                ],
                "conv_kernel_num": 32,
                "maxpool_kernel_size": 3,
                "densed_size": 64,
                "num_layers": 1,
                "learning_rate": 0.01
            },
            "pytorch_cnn_hybrid_12": {
                "matrix_size": [
                    292,
                    292
                ],
                "conv_kernel_num": 32,
                "maxpool_kernel_size": 2,
                "densed_size": 128,
                "num_layers": 2,
                "learning_rate": 0.01
            },
            "pytorch_cnn_hybrid_13": {
                "matrix_size": [
                    292,
                    292
                ],
                "conv_kernel_num": 32,
                "maxpool_kernel_size": 2,
                "densed_size": 128,
                "num_layers": 1,
                "learning_rate": 0.01
            },
            "pytorch_cnn_hybrid_14": {
                "matrix_size": [
                    292,
                    292
                ],
                "conv_kernel_num": 32,
                "maxpool_kernel_size": 2,
                "densed_size": 64,
                "num_layers": 2,
                "learning_rate": 0.01
            },
            "pytorch_cnn_hybrid_15": {
                "matrix_size": [
                    292,
                    292
                ],
                "conv_kernel_num": 32,
                "maxpool_kernel_size": 2,
                "densed_size": 64,
                "num_layers": 1,
                "learning_rate": 0.01
            }
        }
    }
}
