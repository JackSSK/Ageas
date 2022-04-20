{
  "SVM": {
      "kernel": [
          "linear"
      ],
      "gamma": [
          "scale",
          "auto"
      ],
      "C": [
          1.0
      ],
      "degree": [
          3
      ],
      "cache_size": [
          500
      ],
      "probability": [
          true
      ]
  },
  "GBM": {
      "boosters": [
          "gbtree"
      ],
      "objectives": [
          "multi:softmax",
          "binary:logistic"
      ],
      "num_class":[
          2
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
  },
  "CNN_Hybrid":{
      "epoch": 2,
      "batch_size": [
          5,
          10
      ],
      "matrix_size": [
          [
              292,
              292
          ]
      ],
      "conv_kernel_num": [
          32,
          64
      ],
      "maxpool_kernel_size": [
          2,
          3
      ],
      "densed_size": [
          64,
          128
      ],
      "num_layers": [
          1,
          2
      ],
      "learning_rate": [
          0.01
      ]
  },
  "CNN_1D": {
      "epoch": 2,
      "batch_size": [
          5,
          10
      ],
      "conv_kernel_size": [
          32
      ],
      "conv_kernel_num": [
          32,
          64
      ],
      "maxpool_kernel_size": [
          2,
          3
      ],
      "densed_size": [
          64,
          128
      ],
      "num_layers": [
          1,
          2
      ],
      "learning_rate": [
          0.01
      ]
  }
}
