{
    "SVM": {
        "kernels": [
            "linear"
        ],
        "gammas": [
            "scale",
            "auto"
        ],
        "Cs": [
            1.0
        ],
        "degrees": [
            3
        ]
    },
    "CNN": {
        "epochs": 2,
        "batchSizes": [
            5,
            10
        ],
        "layerSetLimit": true,
        "matrixSizes": [
            [
                292,
                292
            ]
        ],
        "convKernelSizes": [
            32
        ],
        "convKernelNums": [
            32,
            64
        ],
        "modelTypes": [
            "1d",
            "hybrid"
        ],
        "maxPoolKernelSizes": [
            2,
            3
        ],
        "densedSizes": [
            64,
            128
        ],
        "layerNums": [
            1,
            2
        ],
        "learningRatios": [
            0.01
        ]
    },
    "XGB": {
        "boosters": [
            "gbtree"
        ],
        "objectives": [
            "multi:softmax",
            "binary:logistic"
        ],
        "eval_metric": [
            "mlogloss"
        ],
        "etas": [
            0.1,
            0.3
        ],
        "gammas": [
            0,
            0.1
        ],
        "max_depths": [
            6
        ],
        "min_child_weights": [
            1,
            2
        ],
        "alphas": [
            0
        ]
    }
}
