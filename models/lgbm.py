from lgbm.config import config

import numpy as np

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


