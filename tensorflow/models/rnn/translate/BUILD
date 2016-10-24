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
        "//tensorflow/models/rnn/translate/utils:data_utils",
        "//tensorflow/models/rnn/translate/utils:train_utils",
        "//tensorflow/models/rnn/translate/utils:model_utils",
        "//tensorflow/models/rnn/translate/seq2seq:seq2seq_model",
        "//tensorflow/models/rnn/translate/seq2seq:engine",
        "//tensorflow/models/rnn/translate/seq2seq:tf_seq2seq",
        "//tensorflow/models/rnn/translate/seq2seq:wrapper_cells",
	":train",
	":decode",
    ],
)

py_binary(
    name = "train",
    srcs = [
        "train.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/models/rnn/translate/utils:data_utils",
        "//tensorflow/models/rnn/translate/utils:train_utils",
        "//tensorflow/models/rnn/translate/utils:model_utils",
        "//tensorflow:tensorflow_py",
    ],
)

py_binary(
    name = "decode",
    srcs = [
        "decode.py",
    ],
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/models/rnn/translate:train",
        "//tensorflow/models/rnn/translate/utils:data_utils",
        "//tensorflow/models/rnn/translate/utils:model_utils",
        "//tensorflow:tensorflow_py",
    ],
)

py_test(
    name = "translate_test",
    size = "medium",
    srcs = [
        "train.py",
    ],
    args = [
        "--self_test=True",
    ],
    main = "train.py",
    srcs_version = "PY2AND3",
    deps = [
        "//tensorflow/models/rnn/translate/utils:data_utils",
        "//tensorflow/models/rnn/translate/utils:train_utils",
        "//tensorflow/models/rnn/translate/utils:model_utils",
        "//tensorflow:tensorflow_py",
    ],
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
