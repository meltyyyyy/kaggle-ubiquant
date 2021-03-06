import pandas as pd

from logging import Formatter, FileHandler, getLogger

LOG_DIR = '../logs/'
TRAIN_DATA = '~/ubiquant/data/input/train.csv'
TRAIN_FEATHER_DATA = '~/ubiquant/data/input/train.feather'
TEST_DATA = '~/ubiquant/data/input/test.csv'
TEST_FEATHER_DATA = '~/ubiquant/data/input/test.feather'

logger = getLogger(__name__)
log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')

handler = FileHandler(LOG_DIR + 'load_data.py.log',mode='w')
handler.setLevel('DEBUG')
handler.setFormatter(log_fmt)
logger.setLevel('DEBUG')
logger.addHandler(handler)

def read_csv(path):
    logger.debug('enter')
    df = pd.read_csv(path)
    logger.debug('exit')
    return df

def load_train_feather():
    logger.info('enter')
    df = pd.read_feather(TRAIN_FEATHER_DATA)
    logger.info('exit')
    return df

def load_train_data():
    data_types_dict = {
    'time_id': 'int32',
    'investment_id': 'int16',
    "target": 'float16',
    }

    features = [f'f_{i}' for i in range(300)]

    for f in features:
        data_types_dict[f] = 'float16'

    logger.debug('enter')
    df = pd.read_csv(TRAIN_DATA,
                usecols = data_types_dict.keys(),
                dtype=data_types_dict,
                index_col = 0)

    logger.debug('exit')
    return df

def load_test_feather():
    logger.debug('enter')
    df = pd.read_feather(TEST_FEATHER_DATA)
    logger.debug('exit')
    return df

def load_test_data():
    logger.debug('enter')
    df = pd.read_csv(TEST_DATA)
    logger.debug('exit')
    return df

if __name__ == '__main__':
    print(load_train_data().head())
    print(load_test_data().head())
