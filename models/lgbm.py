import sys, os
sys.path.append(os.pardir)

from configs.lgbm_config import config
from utils.load_data import load_train_data

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from logging import StreamHandler, DEBUG, Formatter, FileHandler, getLogger

LOG_DIR = '../logs/'

logger = getLogger(__name__)
log_fmt = Formatter('%(asctime)s %(name)s %(lineno)d [%(levelname)s][%(funcName)s] %(message)s ')

handler = StreamHandler()
handler.setLevel('INFO')
handler.setFormatter(log_fmt)
logger.addHandler(handler)

handler = FileHandler(LOG_DIR + 'lgbm.py.log',mode='w')
handler.setLevel(DEBUG)
handler.setFormatter(log_fmt)
logger.setLevel('DEBUG')
logger.addHandler(handler)

logger.info('start')

lgb_params = {
    'objective': 'regression',
    'n_estimators': config.EPOCHS,
    'random_state': config.SEED,
    'learning_rate': config.LR,
    'num_leaves': 2 ** np.random.randint(3,8),
    'subsample': np.random.uniform(0.5,1.0),
    'subsample_freq': 1,
    'n_jobs': -1,
    'min_child_samples': 1000,
#     'device':'gpu',
    "metric":"rmse"
}

# load data
logger.info('start loading')
train_df = load_train_data()
logger.info('train data loaded: {}'.format(train_df.shape))

features = [f'f_{i}' for i in range(300)]
lgb_oof = np.zeros(train_df.shape[0])
val_pred = np.zeros(train_df.shape[0])
lgb_importances = pd.DataFrame()
models = []
skf = StratifiedKFold(n_splits=config.FOLDS, shuffle = True , random_state=config.SEED)

for fold, (trn_idx, val_idx) in enumerate(skf.split(X=train_df, y=train_df['investment_id'])):
	logger.info('fold: {}'.format(fold))
	train = train_df.iloc[trn_idx]
	valid = train_df.iloc[val_idx]

	logger.info('params: {}'.format(lgb_params))
	model = LGBMRegressor(**lgb_params)

	logger.info('training start')
	model.fit(
	train[features],
	train[config.TARGET],
	eval_set = (valid[features],valid[config.TARGET]),
	early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
	verbose=config.VERBOSE,
	)
	logger.info('training finished')

	fi_tmp = pd.DataFrame()
	fi_tmp['feature'] = model.feature_name_
	fi_tmp['importance'] = model.feature_importances_
	fi_tmp['fold'] = fold
	fi_tmp['seed'] = config.SEED
	lgb_importances = lgb_importances.append(fi_tmp)

	val_pred = model.predict(valid[features])
	logger.debug('val_pred: {}'.format(val_pred))

	rmse = np.sqrt(mean_squared_error(valid[config.TARGET], val_pred))
	logger.debug('fold: {}, rmse: {:.6f}, elapsed time: {:.2f}sec'.format(fold,rmse,elapsed))
	models.append(model)

logger.info('end')

