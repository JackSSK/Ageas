{
  "RFC": {
      "n_estimators": [
          100,
          50
      ],
      "criterion": [
        "gini",
        "entropy"
      ],
      "max_features":[
        0.01,
        "auto",
        "log2",
        null
      ]
  },
  "GNB": {
      "var_smoothing": [
          1e-9,
          1e-6,
          1e-3,
          1
      ]
  },
  "SVM": {
      "kernel": [
          "rbf",
          "poly",
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
      "booster": [
          "gbtree"
      ],
      "objective": [
          "multi:softmax",
          "binary:logistic"
      ],
      "eval_metric": [
          "mlogloss"
      ],
      "eta": [
          0.1,
          0.3
      ],
      "gamma": [
          0,
          0.1
      ],
      "max_depth": [
          6
      ],
      "min_child_weight": [
          1,
          2
      ],
      "alpha": [
          0
      ]
  },
  "CNN_Hybrid":{
      "epoch": [
        2
      ],
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
          0.1
      ]
  },
  "CNN_1D": {
      "epoch": [
        2
      ],
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
          0.1
      ]
  },
  "RNN": {
      "epoch": [
        2
      ],
      "batch_size": [
          5,
          10
      ],
      "hidden_size":[
          128,
          256
      ],
      "num_layers":[
          2,
          3
      ],
      "learning_rate":[
          0.01
      ],
      "dropout_prob":[
          0.2
      ]
  },
  "LSTM": {
      "epoch": [
        2
      ],
      "batch_size": [
          5,
          10
      ],
      "hidden_size":[
          128,
          256
      ],
      "num_layers":[
          2,
          3
      ],
      "learning_rate":[
          0.01
      ],
      "dropout_prob":[
          0.2
      ]
  },
  "GRU": {
      "epoch": [
        2
      ],
      "batch_size": [
          5,
          10
      ],
      "hidden_size":[
          128,
          256
      ],
      "num_layers":[
          2,
          3
      ],
      "learning_rate":[
          0.01
      ],
      "dropout_prob":[
          0.2
      ]
  }
}
