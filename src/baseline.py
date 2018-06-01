from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from util import setup_logs
import data

logger = setup_logs("", "../tmp/tmp.log")


def validate():
    logger.info("loading data")
    X, y = data.load()

    train_rows = y.shape[0]
    X = X[:train_rows]
    print(X.shape)

    import lightgbm as lgb

    params = {
        'boosting_type': 'gbdt',
        'objective': 'regression',
        'metric': 'rmse',
        'num_leaves': 31,
        'learning_rate': 0.1,
        # 'feature_fraction': 0.5,
        # 'bagging_fraction': 0.75,        
        'verbose': 0,
    }
    
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.20, random_state=23)
    
    lgb_train = lgb.Dataset(
        X_train,
        y_train,
        # categorical_feature=[idx for idx, name in enumerate(categorical)],
        free_raw_data=False,
    )
    
    lgb_valid = lgb.Dataset(X_valid, y_valid, reference=lgb_train)

    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=3000,
        valid_sets=[lgb_train, lgb_valid],
        early_stopping_rounds=100,
        verbose_eval=50,
    )

    print('rmse: ', np.sqrt())
    print('rmse:', np.sqrt(mean_squared_error(y_valid, lgb_clf.predict(X_valid))))


if __name__ == "__main__":
    validate()
