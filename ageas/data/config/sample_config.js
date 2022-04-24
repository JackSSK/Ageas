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
            "booster": "gbtree",
            "objective": "binary:logistic",
            "eval_metric": "mlogloss",
            "eta": 0.3,
            "gamma": 0.1,
            "max_depth": 6,
            "min_child_weight": 2,
            "alpha": 0
        },
        "xgboost_gbm_1": {
            "booster": "gbtree",
            "objective": "binary:logistic",
            "eval_metric": "mlogloss",
            "eta": 0.3,
            "gamma": 0.1,
            "max_depth": 6,
            "min_child_weight": 1,
            "alpha": 0
        },
        "xgboost_gbm_2": {
            "booster": "gbtree",
            "objective": "binary:logistic",
            "eval_metric": "mlogloss",
            "eta": 0.3,
            "gamma": 0,
            "max_depth": 6,
            "min_child_weight": 2,
            "alpha": 0
        },
        "xgboost_gbm_3": {
            "booster": "gbtree",
            "objective": "binary:logistic",
            "eval_metric": "mlogloss",
            "eta": 0.3,
            "gamma": 0,
            "max_depth": 6,
            "min_child_weight": 1,
            "alpha": 0
        },
        "xgboost_gbm_4": {
            "booster": "gbtree",
            "objective": "binary:logistic",
            "eval_metric": "mlogloss",
            "eta": 0.1,
            "gamma": 0.1,
            "max_depth": 6,
            "min_child_weight": 2,
            "alpha": 0
        },
        "xgboost_gbm_5": {
            "booster": "gbtree",
            "objective": "binary:logistic",
            "eval_metric": "mlogloss",
            "eta": 0.1,
            "gamma": 0.1,
            "max_depth": 6,
            "min_child_weight": 1,
            "alpha": 0
        },
        "xgboost_gbm_6": {
            "booster": "gbtree",
            "objective": "binary:logistic",
            "eval_metric": "mlogloss",
            "eta": 0.1,
            "gamma": 0,
            "max_depth": 6,
            "min_child_weight": 2,
            "alpha": 0
        },
        "xgboost_gbm_7": {
            "booster": "gbtree",
            "objective": "binary:logistic",
            "eval_metric": "mlogloss",
            "eta": 0.1,
            "gamma": 0,
            "max_depth": 6,
            "min_child_weight": 1,
            "alpha": 0
        },
        "xgboost_gbm_8": {
            "booster": "gbtree",
            "objective": "multi:softmax",
            "eval_metric": "mlogloss",
            "eta": 0.3,
            "gamma": 0.1,
            "max_depth": 6,
            "min_child_weight": 2,
            "alpha": 0,
            "num_class": 2
        },
        "xgboost_gbm_9": {
            "booster": "gbtree",
            "objective": "multi:softmax",
            "eval_metric": "mlogloss",
            "eta": 0.3,
            "gamma": 0.1,
            "max_depth": 6,
            "min_child_weight": 1,
            "alpha": 0,
            "num_class": 2
        },
        "xgboost_gbm_10": {
            "booster": "gbtree",
            "objective": "multi:softmax",
            "eval_metric": "mlogloss",
            "eta": 0.3,
            "gamma": 0,
            "max_depth": 6,
            "min_child_weight": 2,
            "alpha": 0,
            "num_class": 2
        },
        "xgboost_gbm_11": {
            "booster": "gbtree",
            "objective": "multi:softmax",
            "eval_metric": "mlogloss",
            "eta": 0.3,
            "gamma": 0,
            "max_depth": 6,
            "min_child_weight": 1,
            "alpha": 0,
            "num_class": 2
        },
        "xgboost_gbm_12": {
            "booster": "gbtree",
            "objective": "multi:softmax",
            "eval_metric": "mlogloss",
            "eta": 0.1,
            "gamma": 0.1,
            "max_depth": 6,
            "min_child_weight": 2,
            "alpha": 0,
            "num_class": 2
        },
        "xgboost_gbm_13": {
            "booster": "gbtree",
            "objective": "multi:softmax",
            "eval_metric": "mlogloss",
            "eta": 0.1,
            "gamma": 0.1,
            "max_depth": 6,
            "min_child_weight": 1,
            "alpha": 0,
            "num_class": 2
        },
        "xgboost_gbm_14": {
            "booster": "gbtree",
            "objective": "multi:softmax",
            "eval_metric": "mlogloss",
            "eta": 0.1,
            "gamma": 0,
            "max_depth": 6,
            "min_child_weight": 2,
            "alpha": 0,
            "num_class": 2
        },
        "xgboost_gbm_15": {
            "booster": "gbtree",
            "objective": "multi:softmax",
            "eval_metric": "mlogloss",
            "eta": 0.1,
            "gamma": 0,
            "max_depth": 6,
            "min_child_weight": 1,
            "alpha": 0,
            "num_class": 2
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
                "num_layers": 2,
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
                "num_layers": 1,
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
    },
    "RNN": {
        "Epoch": 2,
        "Batch_Size": [
            5,
            10
        ],
        "RNN": {
            "pytorch_rnn_0": {
                "hidden_size": 256,
                "num_layers": 3,
                "learning_rate": 0.01
            },
            "pytorch_rnn_1": {
                "hidden_size": 256,
                "num_layers": 2,
                "learning_rate": 0.01
            },
            "pytorch_rnn_2": {
                "hidden_size": 128,
                "num_layers": 3,
                "learning_rate": 0.01
            },
            "pytorch_rnn_3": {
                "hidden_size": 128,
                "num_layers": 2,
                "learning_rate": 0.01
            }
        },
        "LSTM": {
            "pytorch_lstm_0": {
                "hidden_size": 256,
                "num_layers": 3,
                "learning_rate": 0.01
            },
            "pytorch_lstm_1": {
                "hidden_size": 256,
                "num_layers": 2,
                "learning_rate": 0.01
            },
            "pytorch_lstm_2": {
                "hidden_size": 128,
                "num_layers": 3,
                "learning_rate": 0.01
            },
            "pytorch_lstm_3": {
                "hidden_size": 128,
                "num_layers": 2,
                "learning_rate": 0.01
            }
        },
        "GRU": {
            "pytorch_gru_0": {
                "hidden_size": 256,
                "num_layers": 3,
                "learning_rate": 0.01
            },
            "pytorch_gru_1": {
                "hidden_size": 256,
                "num_layers": 2,
                "learning_rate": 0.01
            },
            "pytorch_gru_2": {
                "hidden_size": 128,
                "num_layers": 3,
                "learning_rate": 0.01
            },
            "pytorch_gru_3": {
                "hidden_size": 128,
                "num_layers": 2,
                "learning_rate": 0.01
            }
        }
    }
}
