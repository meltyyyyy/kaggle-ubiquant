import sys, os
sys.path.append(os.pardir)

from configs.lgbm_config import config
from utils.load_data import load_train_data

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMRegressor

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
train_df = load_train_data()
features = [f'f_{i}' for i in range(300)]

lgb_oof = np.zeros(train_df.shape[0])
val_pred = np.zeros(train_df.shape[0])
lgb_importances = pd.DataFrame()
models = []
skf = StratifiedKFold(n_splits=config.FOLDS, shuffle = True , random_state=config.SEED)

for fold, (trn_idx, val_idx) in enumerate(skf.split(X=train_df, y=train_df['investment_id'])):
	print(f"===== fold {fold} =====")
	train = train_df.iloc[trn_idx]
	valid = train_df.iloc[val_idx]

	model = LGBMRegressor(**lgb_params)
	model.fit(
	train[features],
	train[config.TARGET],
	eval_set = (valid[features],valid[config.TARGET]),
	early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
	verbose=config.VERBOSE,
	)

	fi_tmp = pd.DataFrame()
	fi_tmp['feature'] = model.feature_name_
	fi_tmp['importance'] = model.feature_importances_
	fi_tmp['fold'] = fold
	fi_tmp['seed'] = config.SEED
	lgb_importances = lgb_importances.append(fi_tmp)

	val_pred = model.predict(valid[features])

	rmse = np.sqrt(mean_squared_error(valid[config.TARGET], val_pred))
	print(f"fold {fold} - lgb rmse: {rmse:.6f}, elapsed time: {elapsed:.2f}sec\n")
	models.append(model)



