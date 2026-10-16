"""공통 경로와 CPU 학습·추론에 필요한 TensorFlow/Keras 설정."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
BREEDS = ["Birman", "Ragdoll", "Siamese", "Beagle", "Pug", "Shiba Inu"]

os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ.setdefault("KERAS_HOME", str(ROOT / ".cache/keras"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras
import numpy as np
import tensorflow as tf

tf.get_logger().setLevel("ERROR")

# 사용 가능한 GPU가 있어도 이 실습의 학습과 추론은 CPU에서 실행한다.
tf.config.set_visible_devices([], "GPU")
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(2)
