import sys, os
sys.path.append(os.pardir)

from utils.load_data import load_train_feather
from logging import Formatter, FileHandler, getLogger
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras

LOG_DIR = '../logs/'

logger = getLogger(__name__)
log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s')

handler = FileHandler(LOG_DIR + 'dnn.py.log',mode='w')
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
    investment_id_lookup_layer = layers.IntegerLookup(max_tokens=investment_ids_size)
    investment_id_lookup_layer.adapt(investment_ids)

