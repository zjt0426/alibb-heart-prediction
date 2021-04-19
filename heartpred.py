import pandas as pd
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import joblib




scoring = 'accuracy'




data = pd.read_csv(r'D:\Desktop\heartpredict\train.csv')
kind = data.groupby('label').count()
X = data['heartbeat_signals'].values
X = X.reshape(100000, -1)
train_data = []
for i in X:
    for j in i:
        j = j.split(',')
        k = list(map(float, j))
        train_data.append(k)
X = train_data
Y = data.label.values
X0, X1,  X2, X3 = [], [], [], []
for i, j in enumerate(Y):  # 根据 labels 的值获得不同 labels 下的 index
    if j == 0:
        X0.append(i)
    elif j == 1:
        X1.append(i)
    elif j == 2:
        X2.append(i)
    else:
        X3.append(i)
s0 = random.sample(X0, 3562)
s1 = random.sample(X1, 3562)
s2 = random.sample(X2, 3562)
s3 = random.sample(X3, 3562)
S = s0 + s1 + s2 + s3
Y_ = []
for i in range(4):
    y = [i] * 3562
    Y_ += y
X_ = []
for j in S:
    X_.append(X[j])
fulldata = np.array(X_)
fulllabel = np.array(Y_)

data_num, _ = fulldata.shape
index = np.arange(data_num)
np.random.shuffle(index)
X_train = fulldata[index]
y_train = fulllabel[index]
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=7)


def linearmodels(*args):
    models ={}
    models['LR'] = LogisticRegression(max_iter=4000)
    models['LDA'] = LinearDiscriminantAnalysis()
    models['KNN'] = KNeighborsClassifier(n_neighbors=1)
    models['CART'] = DecisionTreeClassifier()
    models['NB'] = GaussianNB()
    models['SVM'] = SVC()

    result = []

    for key in models:
        kfold = KFold(n_splits=10, shuffle=False)
        cv_result = cross_val_score(models[key], X_train, y_train, cv=kfold, scoring=scoring)
        result.append(cv_result)

        print('%s: %f (%f)' % (key, cv_result.mean(), cv_result.std()))


    fig = plt.figure(figsize=(10, 8))
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(result)
    ax.set_xticklabels(models.keys())
    plt.savefig(r'D:\Desktop\heartpredict\boxplot.png')
    plt.show()

def StanderScaler(*args):
    pipelines = {}
    pipelines['ScalerLR'] = Pipeline([('Scaler', StandardScaler()), ('LR', LogisticRegression())])
    pipelines['ScalerLASSO'] = Pipeline([('Scaler', StandardScaler()), ('LDA', LinearDiscriminantAnalysis())])
    pipelines['ScalerEN'] = Pipeline([('Scaler', StandardScaler()), ('KNN', KNeighborsClassifier(n_neighbors=1))])
    pipelines['ScalerKNN'] = Pipeline([('Scaler', StandardScaler()), ('CART', DecisionTreeClassifier())])
    pipelines['ScalerTREE'] = Pipeline([('Scaler', StandardScaler()), ('NB', GaussianNB())])
    pipelines['ScalerSVM'] = Pipeline([('Scaler', StandardScaler()), ('SVM', SVC())])

    results_ = []
    for key in pipelines:
        kfold = KFold(n_splits=10, shuffle=False)
        pip_result = cross_val_score(pipelines[key], X_train, y_train, cv=kfold, scoring=scoring)
        results_.append(pip_result)
        print('Stand_results %s : %f (%f)' % (key, pip_result.mean(), pip_result.std()))



def KNN_optim(*args):    # best is 1
    scaler = StandardScaler().fit(X_train)
    rescaledX = scaler.transform(X_train)
    param_grid = {'n_neighbors': [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]}
    model = KNeighborsClassifier()
    kfold = KFold(n_splits=10, shuffle=False)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid_result = grid.fit(rescaledX, y_train)
    print('best param is : %s, and score is : %f' % (grid_result.best_params_, grid_result.best_score_))
    cv_results = zip(grid_result.cv_results_['mean_test_score'], grid_result.cv_results_['std_test_score'], grid_result.cv_results_['params'])
    for mean, std, param in cv_results:
        print(param, mean, std)

def SVM_optim():
    scaler = StandardScaler().fit(X_train)
    rescalerdX = scaler.transform(X_train).astype(float)
    param_grid = {}
    param_grid['C'] = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0, 1.3, 1.5, 1.7, 2.0]
    param_grid['kernel'] = ['linear', 'poly', 'rbf', 'sigmoid']    # 'precomputed'成对
    model = SVC()
    kfold = KFold(n_splits=10, shuffle=False)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid_result = grid.fit(rescalerdX, y_train)
    print('best param is : %s, and score is : %f' % (grid_result.best_params_, grid_result.best_score_))
    cv_results = zip(grid_result.cv_results_['mean_test_score'], grid_result.cv_results_['std_test_score'],
                     grid_result.cv_results_['params'])
    for mean, std, param in cv_results:
        print(param, mean, std)



def ensem(*args):
    ensembles = {}
    ensembles['ScalerAB'] = Pipeline([('Scaler', StandardScaler()), ('AB', AdaBoostClassifier())])
    # ensembles['ScalerAB-KNN'] = Pipeline([('Scaler', StandardScaler()), ('ABKNN', AdaBoostRegressor(base_estimator=KNeighborsRegressor(n_neighbors=1)))])
    # ensembles['ScalerAB-LR'] = Pipeline([('Scaler', StandardScaler()), ('ABLR', AdaBoostRegressor(LinearRegression()))])
    ensembles['ScalerRFR'] = Pipeline([('Scaler', StandardScaler()), ('RFR', RandomForestClassifier())])
    ensembles['ScalerGBM'] = Pipeline([('Scaler', StandardScaler()), ('RFR', GradientBoostingClassifier())])
    ensembles['ScalerETR'] = Pipeline([('Scaler', StandardScaler()), ('ETR', ExtraTreesClassifier())])
    # ensembles['ScalerRBR'] = Pipeline([('Scaler', StandardScaler()), ('RBR', GradientBoostingRegressor())])


    results__ = []

    for key in ensembles:
        kfold = KFold(n_splits=10, shuffle=False)
        cv__result = cross_val_score(ensembles[key], X_train, y_train, scoring=scoring, cv=kfold)
        results__.append(cv__result)

        print('%s: %f (%f)' % (key, cv__result.mean(), cv__result.std()))
def fitting1():
    # 集成算法调参随机梯度和极端梯度
    scaler = StandardScaler().fit(X_train)
    rescalerdX = scaler.transform(X_train)
    param_grid = {'n_estimators': [10, 100, 150, 200]}
    model = RandomForestClassifier()
    kfold = KFold(n_splits=10, shuffle=False)
    grid = GridSearchCV(model, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid__result = grid.fit(rescalerdX, y_train)

    print('%s : %f ' % (grid__result.best_params_, grid__result.best_score_))

def fitting2(self):
    scaler = StandardScaler().fit(X_train)
    rescalerdX = scaler.transform(X_train)
    param_grid = {'n_estimators': [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]}
    model = ExtraTreesClassifier()
    kfold = KFold(n_splits=10, shuffle=False)
    grid = GridSearchCV(model, param_grid=param_grid, scoring=scoring, cv=kfold)
    grid__result = grid.fit(rescalerdX, y_train)

    print('%s : %f ' % (grid__result.best_params_, grid__result.best_score_))


model_file = r'C:\Users\zjt\PycharmProjects\pythonProject\finalized_model_joblib.sav'
def final_model():
    scaler = StandardScaler().fit(X_train)
    rescaledX = scaler.transform(X_train)
    # model = RandomForestClassifier(n_estimators=100)
    model = KNeighborsClassifier(n_neighbors=1)
    model.fit(rescaledX, y_train)

    rescaledX_test = scaler.transform(X_test)
    predictions = model.predict(rescaledX_test)
    # print(mean_squared_error(y_test, predictions))
    print(accuracy_score(y_test, predictions))
    print(confusion_matrix(y_test, predictions))
    print(classification_report(y_test, predictions))
    # print(predictions)
    joblib.dump(value=model, filename=model_file, compress=True)

def test_data_model():
    data0 = pd.read_csv(r'D:\Desktop\heartpredict\testA.csv')
    # print(data.head(3))

    Signals = data0.heartbeat_signals.values
    Signals = Signals.reshape(20000, -1)
    # print(X)

    test_data = []
    for i in Signals:
        for j in i:
            j = j.split(',')
            k = list(map(float, j))
            test_data.append(k)

    X_test = test_data
    X_test = np.asarray(X_test)

    model = joblib.load(filename=model_file)
    # print("model has loaded!")
    # print(type(model))
    scaler = StandardScaler().fit(X_train)
    # rescaledX = scaler.transform(X_train)

    rescaledX_test = scaler.transform(X_test)
    predictions = model.predict(rescaledX_test)
    pre_prob = model.predict_proba(rescaledX_test)
    # print(predictions)
    # print(len(predictions))
    # print(type(predictions))
    # print(pre_prob)
    # print(pre_prob.shape)
    # final_data = pd.DataFrame(data=pre_prob,index=)
    # pre_prob.to_csv(r'D:\Desktop\heartpredict\savetest.csv')
    final_data = pd.DataFrame(data=pre_prob, index=range(100000, 120000), columns=['label_0', 'label_1', 'label_2', 'label_3'])
    final_data.to_csv(r'D:\Desktop\heartpredict\savetest.csv')
    # np.savetxt(r'D:\Desktop\heartpredict\savetest.csv', pre_prob, delimiter=',', fmt='%f')

if __name__ == '__main__':
    # ensem()
    # linearmodels()
    # KNN_optim()
    # SVM_optim()
    # fitting1()
    final_model()
    test_data_model()
