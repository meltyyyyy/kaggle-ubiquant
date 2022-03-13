import os, sys
sys.path.append(os.pardir)
from utils.get_model_summary import get_model_summary
from utils.load_data import load_train_feather
from tensorflow.python.ops import math_ops
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
from logging import Formatter, FileHandler, getLogger


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


def preprocess(item):
    return (item["investment_id"], item["features"]), item["target"]


def make_dataset(df, batch_size=4096, mode="train"):
    ds = tf.data.Dataset.from_tensor_slices(df)
    ds = ds.map(decode_function)
    ds = ds.map(preprocess)
    if mode == "train":
        ds = ds.shuffle(batch_size * 4)
    ds = ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
    return ds


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


def get_model():
    investment_id_inputs = tf.keras.Input((1, ), dtype=tf.uint16)
    features_inputs = tf.keras.Input((300, ), dtype=tf.float16)

    investment_id_x = investment_id_lookup_layer(investment_id_inputs)
    investment_id_x = layers.Embedding(
        investment_ids_size, 32, input_length=1)(investment_id_x)
    investment_id_x = layers.Reshape((-1, ))(investment_id_x)
    investment_id_x = layers.Dense(64, activation='swish')(investment_id_x)
    investment_id_x = layers.Dropout(0.1)(investment_id_x)
    investment_id_x = layers.Dense(64, activation='swish')(investment_id_x)
    investment_id_x = layers.Dropout(0.1)(investment_id_x)
    investment_id_x = layers.Dense(64, activation='swish')(investment_id_x)
    investment_id_x = layers.Dropout(0.1)(investment_id_x)

    feature_x = layers.Dense(256, activation='swish')(features_inputs)
    feature_x = layers.Dropout(0.1)(feature_x)
    feature_x = layers.Dense(256, activation='swish')(feature_x)
    feature_x = layers.Dropout(0.1)(feature_x)
    feature_x = layers.Dense(256, activation='swish')(feature_x)
    feature_x = layers.Dropout(0.1)(feature_x)

    x = layers.Concatenate(axis=1)([investment_id_x, feature_x])
    x = layers.Dense(512, activation='swish', kernel_regularizer="l2")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(128, activation='swish', kernel_regularizer="l2")(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(32, activation='swish', kernel_regularizer="l2")(x)
    x = layers.Dropout(0.1)(x)
    output = layers.Dense(1)(x)
    rmse = keras.metrics.RootMeanSquaredError(name="rmse")
    model = tf.keras.Model(
        inputs=[investment_id_inputs, features_inputs], outputs=[output])
    model.compile(optimizer=tf.optimizers.Adam(0.001), loss='mse',
                  metrics=['mse', "mae", "mape", rmse, correlation])
    return model


logger.info('build model')
model = get_model()
logger.info('model summery:\n{}'.format(get_model_summary(model)))
keras.utils.plot_model(model, show_shapes=True)

models = []
for i in range(5):
    
