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
        ":seq2seq_model",
        ":engine",
        ":tf_seq2seq",
        ":wrapper_cells"
    ],
)

py_library(
    name = "seq2seq_model",
    srcs = [
        "seq2seq_model.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/models/rnn/translate/utils:data_utils",
	":wrapper_cells",
        "//tensorflow:tensorflow_py",
        "//tensorflow/models/rnn:seq2seq",
    ],
)

py_library(
    name = "engine",
    srcs = [
        "engine.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/models/rnn/translate/utils:data_utils",
        "//tensorflow:tensorflow_py"
    ],
)

py_library(
    name = "tf_seq2seq",
    srcs = [
        "tf_seq2seq.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/models/rnn/translate/utils:data_utils",
	":engine",
	":seq2seq_model",
	":wrapper_cells",
        "//tensorflow:tensorflow_py",
        "//tensorflow/models/rnn:seq2seq",
    ],
)

py_library(
    name = "wrapper_cells",
    srcs = [
        "wrapper_cells.py",
    ],
    srcs_version = "PY2AND3"
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
