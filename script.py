import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.manifold import Isomap
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import IterativeImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
import xgboost as xgb
def cMSE(y_hat, y, c):
    err = y - y_hat
    err = (1 - c) * err ** 2 + c * np.maximum(0, err) ** 2
    return np.sum(err) / err.shape[0]

def calc_fold(X,Y, c, train_ix,valid_ix, pipe):
    pipe.fit(X[train_ix], Y[train_ix])

    trainPred = pipe.predict(X[train_ix])
    trainError = cMSE(trainPred, Y[train_ix], c[train_ix])
    pred = pipe.predict(X[valid_ix])
    error = cMSE(pred, Y[valid_ix], c[valid_ix])
    return error, trainError

data = pd.read_csv('./train_data.csv', index_col=0)

inputers = [KNNImputer(n_neighbors=8, weights="uniform"), SimpleImputer(), IterativeImputer()]
models = [LinearRegression(), Ridge(alpha=10), KNeighborsRegressor(n_neighbors=7)]

testData = pd.read_csv('./test_data.csv', index_col=0).to_numpy()

x = data.drop(columns=['SurvivalTime', 'Censored']).to_numpy()

labeled_data = data.dropna(subset=['SurvivalTime'])

y = labeled_data['SurvivalTime'].to_numpy()
c = labeled_data['Censored'].to_numpy()
X = labeled_data.drop(columns=['SurvivalTime', 'Censored']).to_numpy()

best_err = 100
best_n = 0
best_p = 0
best_alpha = 0
for nn in range(1, 8):
    for nc in range(1, 8):
        for a in range(1, 10):
            alpha = a/10.0
            pipe = make_pipeline( KNNImputer(n_neighbors=nn, weights="uniform"), StandardScaler(), PCA(n_components=nc),
                                 Ridge(alpha=alpha))

            n_folds = 12
            kf = KFold(n_splits=n_folds)
            error = 0
            trainError = 0
            for train_ix, valid_ix in kf.split(X):
                err, trainErr = calc_fold(X, y, c, train_ix, valid_ix, pipe)
                error = error + err
                trainError = trainError + trainErr

            error = error / n_folds
            trainError = trainError / n_folds
            if error < best_err:
                best_err = error
                best_n = nn
                best_p = nc
                best_alpha = alpha
                test = pipe.predict(testData)
                t = pd.DataFrame({'TARGET': test})
                t.to_csv(f'prediction_{nn}_{nc}_{alpha}.csv')

            print("CV ERROR FOR N_NEIGHBORS: " + str(nn) + " N_COMPONENTS: " + str(nc) + " ALPHA: " + str(alpha))
            print(error)
            print("TRAINING ERROR:")
            print(trainError)



print("BEST ERROR: " + str(best_err))
print("BEST N: " + str(best_n))
print("BEST C: " + str(best_p))
print("BEST ALPHA: " + str(best_alpha))