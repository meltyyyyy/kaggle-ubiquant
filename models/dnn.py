import os, sys
sys.path.append(os.pardir)

from utils.load_data import load_train_feather
from logging import Formatter, FileHandler, getLogger
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.python.ops import math_ops

LOG_DIR = '../logs/'

logger = getLogger(__name__)
log_fmt = Formatter(
    '%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s')

handler = FileHandler(LOG_DIR + 'dnn.py.log', mode='w')
handler.setLevel('DEBUG')
handler.setFormatter(log_fmt)
logger.setLevel('DEBUG')
logger.addHandler(handler)

logger.info('start')

logger.info('start loading')
train_feather_df = load_train_feather()
logger.info('finish loading')
logger.info('data shape: {}'.format(train_feather_df.shape))
logger.info('data sample: \n{}'.format(train_feather_df.head()))

# Create an IntegerLookup layer for investment_id input
investment_ids = train_feather_df['investment_id']
investment_ids_size = len(investment_ids) + 1
with tf.device("cpu"):
    investment_id_lookup_layer = layers.IntegerLookup(
        max_tokens=investment_ids_size)
    investment_id_lookup_layer.adapt(investment_ids)


def decode_function(record_bytes):
    return tf.io.parse_single_example(
        # Data
        record_bytes,
        # Schema
        {
            "features": tf.io.FixedLenFeature([300], dtype=tf.float32),
            "time_id": tf.io.FixedLenFeature([], dtype=tf.int64),
            "investment_id": tf.io.FixedLenFeature([], dtype=tf.int64),
            "target": tf.io.FixedLenFeature([], dtype=tf.float32)
        }
    )


def correlation(x, y, axis=-2):
    """Metric returning the Pearson correlation coefficient of two tensors over some axis, default -2."""
    x = tf.convert_to_tensor(x)
    y = math_ops.cast(y, x.dtype)
    n = tf.cast(tf.shape(x)[axis], x.dtype)
    xsum = tf.reduce_sum(x, axis=axis)
    ysum = tf.reduce_sum(y, axis=axis)
    xmean = xsum / n
    ymean = ysum / n
    xvar = tf.reduce_sum(tf.math.squared_difference(x, xmean), axis=axis)
    yvar = tf.reduce_sum(tf.math.squared_difference(y, ymean), axis=axis)
    cov = tf.reduce_sum((x - xmean) * (y - ymean), axis=axis)
    corr = cov / tf.sqrt(xvar * yvar)
    return tf.constant(1.0, dtype=x.dtype) - corr



