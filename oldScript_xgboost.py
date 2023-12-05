import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import optuna
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import Isomap
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import ShuffleSplit
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
import xgboost as xgb

def objective(trial):
    params = {'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 1.0),
              'aft_loss_distribution': trial.suggest_categorical('aft_loss_distribution',
                                                                  ['normal', 'logistic', 'extreme']),
              'aft_loss_distribution_scale': trial.suggest_loguniform('aft_loss_distribution_scale', 0.1, 10.0),
              'max_depth': trial.suggest_int('max_depth', 3, 8),
              'lambda': trial.suggest_loguniform('lambda', 1e-8, 1.0),
              'alpha': trial.suggest_loguniform('alpha', 1e-8, 1.0)}  # Search space
    params.update(base_params)
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, 'valid-aft-nloglik')
    bst = xgb.train(params, dtrain, num_boost_round=10000,
                    evals=[(dtrain, 'train'), (dvalid,'valid')],
                    early_stopping_rounds=50, verbose_eval=False, callbacks=[pruning_callback])
    if bst.best_iteration >= 25:
        return bst.best_score
    else:
        return np.inf  # Reject models with < 25 trees
def cMSE(y_hat, y, c):
    err = y - y_hat
    err = (1 - c) * err ** 2 + c * np.maximum(0, err) ** 2
    return np.sum(err) / err.shape[0]

data = pd.read_csv('./train_data.csv', index_col=0)

inputers = [KNNImputer(n_neighbors=5,weights="distance"), SimpleImputer(), IterativeImputer()]
optuna.logging.set_verbosity(optuna.logging.WARNING)
pca = PCA(n_components=5)
# Missing values imputation
for nr in range(1 , 8):
    selector = SelectKBest(score_func=f_classif, k=nr)
    for y in range(3):

        x = data.drop(columns=['SurvivalTime', 'Censored']).to_numpy()
        X = pd.DataFrame(inputers[y].fit_transform(x), columns=['Age', 'Gender', 'Stage', 'GeneticRisk', 'TreatmentType' , 'ComorbidityIndex', 'TreatmentResponse'])
        data[X.columns] = X


        # Split the data into labeled and unlabeled DataFrames
        labeled_data = data.dropna(subset=['SurvivalTime'])
        #unlabeled_data = data[data['SurvivalTime'].isna()]

        labeled_lower_bound = labeled_data['SurvivalTime'].to_numpy()
        labeled_upper_bound = np.where(labeled_data['Censored'] == 1, 100 - labeled_data['Age'], labeled_data['SurvivalTime'])

        labeledX = labeled_data.drop(columns=['SurvivalTime', 'Censored']).to_numpy()
        x = inputers[y].fit_transform(x)
        pca.fit(x)
        labeledX = pca.transform(labeledX)
        scaler = StandardScaler()
        labeledX = scaler.fit_transform(labeledX)

        # For the unlabeled data, we only have features
        rs = ShuffleSplit(n_splits=2, test_size=.5, random_state=0)
        train_indexes, valid_indexes = next(rs.split(labeledX))

        X_train_selected = labeledX[train_indexes, :]
        X_test_selected = labeledX[valid_indexes, :]


        dtrain = xgb.DMatrix(X_train_selected)
        dtrain.set_float_info('label_lower_bound', labeled_lower_bound[train_indexes])
        dtrain.set_float_info('label_upper_bound', labeled_upper_bound[train_indexes])
        dvalid = xgb.DMatrix(X_test_selected)
        dvalid.set_float_info('label_lower_bound', labeled_lower_bound[valid_indexes])
        dvalid.set_float_info('label_upper_bound', labeled_upper_bound[valid_indexes])


        base_params = {'verbosity': 0,
                      'objective': 'survival:aft',
                      'eval_metric': 'aft-nloglik',
                      'tree_method': 'hist'}  # Hyperparameters common to all trials



        # Run hyperparameter search
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=200, show_progress_bar=False)
        #print('Completed hyperparameter tuning with best aft-nloglik = {}.'.format(study.best_trial.value))
        params = {}
        params.update(base_params)
        params.update(study.best_trial.params)

        # Re-run training with the best hyperparameter combination
        #print('Re-running the best trial... params = {}'.format(params))
        bst = xgb.train(params, dtrain, num_boost_round=10000,
                        evals=[(dtrain, 'train'), (dvalid, 'valid')],
                        early_stopping_rounds=50,verbose_eval=False)

        # Run prediction on the validation set
        pred = bst.predict(dvalid)
        df = pd.DataFrame({'TARGET': pred})

        c = np.where(labeled_upper_bound[valid_indexes] != labeled_lower_bound[valid_indexes], 1 , 0)
        error = cMSE(pred, labeled_lower_bound[valid_indexes], c)
        #print(df)
        print(f"Error with inputer:{y} with nrK:{nr}: {error}")


        testData = pd.read_csv('./test_data.csv', index_col=0).to_numpy()
        Xtest = pd.DataFrame(inputers[y].fit_transform(testData),
                         columns=['Age', 'Gender', 'Stage', 'GeneticRisk', 'TreatmentType', 'ComorbidityIndex',
                                  'TreatmentResponse'])
        test = pca.transform(Xtest)
        testf = xgb.DMatrix(test)
        testPred = bst.predict(testf)
        t = pd.DataFrame({'TARGET': testPred})


        t.to_csv(f'prediction_{y}_{nr}.csv')
