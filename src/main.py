import os
from enum import Enum
import mlflow
import lightgbm as lgb
import pandas as pd
import numpy as np
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from hydra.utils import instantiate
import hydra
import warnings
warnings.filterwarnings("ignore")

"""
корявый пример использования mlflow
"""
def run_model(cfg):

    orig_cwd = hydra.utils.get_original_cwd()
    TARGET = cfg.data.target
    ID = cfg.data.ID
    random_state = cfg.data.random_state

    train = pd.read_csv(orig_cwd+Path_to_Files.raw_path.value+'Train.csv')
    test = pd.read_csv(orig_cwd+Path_to_Files.raw_path.value+'Test.csv')

    miss_df = train.isna().sum().sort_values(ascending=False) / train.shape[0]
    col_to_delete = list(
        miss_df[miss_df > cfg.data.threshold_miss].index)

    train.drop(columns=col_to_delete, inplace=True)
    test.drop(columns=col_to_delete, inplace=True)
    test.drop(columns=[ID], inplace=True)

    cat_cols = [col for col in train.columns if (
        train[col].dtype == "O" and col not in [TARGET, ID])]
    num_col = [col for col in train.columns if (
        train[col].dtype != "O" and col not in [TARGET, ID])]
    for col in num_col:
        train[col] = train[col].apply(lambda x: x/(x+1))
        test[col] = test[col].apply(lambda x: x/(x+1))

    train.fillna(cfg.data.missing_values, inplace=True)
    test.fillna(cfg.data.missing_values, inplace=True)

    x_train, x_test, y_train, y_test = train_test_split(
        train.drop(columns=[TARGET, ID]), train[TARGET], stratify=train[TARGET], random_state=random_state)

    enc = instantiate(cfg.enc, cols=cat_cols)
    x_train_enc = enc.fit_transform(x_train, y_train)
    x_test_enc = enc.transform(x_test)
    test_enc = enc.transform(test)
    model = instantiate(cfg.model)

    x_train_enc.to_csv(orig_cwd+Path_to_Files.processed_path.value +
                       'x_train_target_enc.csv')
    x_test_enc.to_csv(orig_cwd+Path_to_Files.processed_path.value +
                      'x_test_enc_target_enc.csv')
    test_enc.to_csv(orig_cwd+Path_to_Files.processed_path.value +
                    'test_enc_target_enc.csv')

    mlflow.set_experiment(cfg.experiments.name_experiment)

    with mlflow.start_run(run_name=cfg.experiments.run_name) as run:
        model.fit(x_train_enc, y_train)
        preds = model.predict(x_test_enc)
        y_pred = np.where(preds > 0.5, 1, 0)
        f1 = f1_score(y_test, y_pred)
        mlflow.log_metric(key="f1_experiment_score", value=f1)
        ss = pd.read_csv(
            orig_cwd+Path_to_Files.raw_path.value+'SampleSubmission.csv')
        predicts = model.predict_proba(test_enc)[:, 1]
        ss.CHURN = predicts
        ss.to_csv(orig_cwd+Path_to_Files.output.value+'subm.csv', index=False)


@hydra.main(config_name='config/experiment_1.yaml')
def run(cfg):
    run_model(cfg)


run()
