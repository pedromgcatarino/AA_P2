import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.manifold import Isomap
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression, Ridge

from sklearn.decomposition import PCA, KernelPCA

def plot_graph(training_errors, validation_errors, xlabel):
    xvals = list(range(1, len(training_errors) + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(xvals, training_errors, label='Training Error', color='blue')
    plt.plot(xvals, validation_errors, label='Cross-Validation Error', color='red')
    plt.title('Training vs Cross-Validation Error')
    plt.xlabel(xlabel)
    plt.ylabel('Censored Mean Squared Error')
    plt.legend()
    plt.grid(True)
    plt.show()


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
testData = pd.read_csv('./test_data.csv', index_col=0).to_numpy()

#inputers = [KNNImputer(n_neighbors=8, weights="uniform"), SimpleImputer(), IterativeImputer()]
#models = [LinearRegression(), Ridge(alpha=10, solver='cholesky', tol=0.00001), KNeighborsRegressor(n_neighbors=7)]

x = data.drop(columns=['SurvivalTime', 'Censored']).to_numpy()

labeled_data = data.dropna(subset=['SurvivalTime'])

y = labeled_data['SurvivalTime'].to_numpy()
c = labeled_data['Censored'].to_numpy()
X = labeled_data.drop(columns=['SurvivalTime', 'Censored']).to_numpy()


best_err = 100
best_nf = 0
best_p = 0
best_alpha = 0
val_errors = []
train_errors = []
lambs = np.linspace(0.0001, 30)
for nf in range(1, 8):

    # SCALING
    scaler = StandardScaler()
    scaler.fit(x)
    X1 = scaler.transform(X)
    testData1 = scaler.transform(testData)

    # IMPUTING
    imputer = KNNImputer(n_neighbors=3, weights="distance")
    imputer.fit(x)
    x1 = imputer.transform(x)
    X2 = imputer.transform(X1)
    testData2 = imputer.transform(testData1)

    # PCA
    Xc = X2 - np.mean(X2, axis=0)
    pca = PCA(n_components=nf)
    # pca.fit(x)
    Xc = pca.fit_transform(Xc)

    Xt = testData2 - np.mean(testData2, axis=0)
    Xt = pca.fit_transform(Xt)

    # Kernel PCA
    Xc1 = X2 - np.mean(X2, axis=0)
    transformer = KernelPCA(n_components=nf, kernel='rbf', eigen_solver='dense')
    # transformer.fit(x)
    Xc1 = transformer.fit_transform(Xc1)

    Xt1 = testData2 - np.mean(testData2, axis=0)
    Xt1 = transformer.fit_transform(Xt1)

    # ISOMAP
    Xc2 = X2 - np.mean(X2, axis=0)
    embedding = Isomap(n_components=nf)
    # embedding.fit(x)
    Xc2 = embedding.fit_transform(Xc2)

    Xt2 = testData2 - np.mean(testData2, axis=0)
    Xt2 = embedding.fit_transform(Xt2)

    # SELECT K BEST
    Xb = np.append(Xc, Xc1, axis=1)
    Xb = np.append(Xb, Xc2, axis=1)

    Xtt = np.append(Xt, Xt1, axis=1)
    Xtt = np.append(Xtt, Xt2, axis=1)

    skb = SelectKBest(k=nf)
    Xb = skb.fit_transform(Xb, y)
    finaltestData = skb.transform(Xtt)


    # CROSS-VALIDATION
    n_folds = 20
    kf = KFold(n_splits=n_folds)
    error = 0
    trainError = 0
    model = LinearRegression()
    for train_ix, valid_ix in kf.split(X):
        err, trainErr = calc_fold(Xb, y, c, train_ix, valid_ix, model)
        error = error + err
        trainError = trainError + trainErr

    error = error / n_folds
    trainError = trainError / n_folds
    val_errors.append(error)
    train_errors.append(trainError)
    if error < best_err:
        best_err = error
        best_nf = nf
        test = model.predict(finaltestData)
        t = pd.DataFrame({'TARGET': test})
        t.to_csv(f'prediction_{nf}.csv')

plot_graph(train_errors, val_errors, "N Features")


print("BEST ERROR: " + str(best_err))
print("BEST NF: " + str(best_nf))
