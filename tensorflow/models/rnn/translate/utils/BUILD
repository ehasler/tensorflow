# Description:
# Example neural translation models.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "package",
    srcs = [
        "__init__.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        ":data_utils",
        ":model_utils",
        ":train_utils"
    ],
)

py_library(
    name = "data_utils",
    srcs = [
        "data_utils.py",
    ],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow:tensorflow_py"],
)

py_library(
    name = "model_utils",
    srcs = [
        "model_utils.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/models/rnn/translate/seq2seq:seq2seq_model",
        "//tensorflow/models/rnn/translate/seq2seq:tf_seq2seq",
        "//tensorflow:tensorflow_py"
],
)

py_library(
    name = "train_utils",
    srcs = [
        "train_utils.py",
    ],
    srcs_version = "PY2AND3",
    deps = ["//tensorflow:tensorflow_py"],
)

filegroup(
    name = "all_files",
    srcs = glob(
        ["**/*"],
        exclude = [
            "**/METADATA",
            "**/OWNERS",
        ],
    ),
    visibility = ["//tensorflow:__subpackages__"],
)
