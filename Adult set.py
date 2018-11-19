import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn import preprocessing, metrics
from sklearn.svm import SVC
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import pickle


pd.set_option('display.max_columns', 15)

df = pd.read_csv('adult data set.txt', sep=', ', engine='python', na_values=["?"])
df.drop(['native-country',], 1,inplace=True)
df.dropna(0, inplace=True)




def impute_attri():
    relation = {'Not-in-family': 0,'Husband': 1,'Wife':2,
                'Own-child':3, 'Unmarried':4,'Other-relative':5}
    workcls = {'State-gov':11, 'Self-emp-not-inc':12, 'Private':13, 'Federal-gov':14, 'Local-gov':15,
               'Self-emp-inc':16, 'Without-pay':17, 'Never-worked':18}
    maritalstat = {'Never-married':20, 'Married-civ-spouse':21, 'Divorced':22, 'Married-spouse-absent':23,
                   'Separated':24, 'Married-AF-spouse':25, 'Widowed':26}
    edu = {'Bachelors':31, 'HS-grad':32, '11th':33, 'Masters':34, '9th':35, 'Some-college':36, 'Assoc-acdm':37,
           'Assoc-voc':38, '7th-8th':39, 'Doctorate':40, 'Prof-school':41, '5th-6th':42, '10th':43,
           'Preschool':44, '12th':45, '1st-4th':46,}
    occ = {'Adm-clerical':50, 'Exec-managerial':51, 'Handlers-cleaners':52, 'Prof-specialty':53,
           'Other-service':54, 'Sales':55, 'Craft-repair':56, 'Transport-moving':57,
           'Farming-fishing':58, 'Machine-op-inspct':59, 'Tech-support':60, 'Protective-serv':61,
           'Armed-Forces':62, 'Priv-house-serv':63,}
    rac = {'White':70, 'Black':71, 'Asian-Pac-Islander':72, 'Amer-Indian-Eskimo':73, 'Other':74}
    df['sex'] = np.where(df['sex'] == 'Male', 100, 101)
    df['relationship'] = [relation[x] for x in df['relationship']]
    df['workclass'] = [workcls[x] for x in df['workclass']]
    df['marital-status'] = [maritalstat[x] for x in df['marital-status']]
    df['education'] = [edu[x] for x in df['education']]
    df['occupation'] = [occ[x] for x in df['occupation']]
    df['race'] = [rac[x] for x in df['race']]
    df['money'] = np.where(df['money'] == '>50K', '>50', '<=50')
    # print(df['money'].unique())
    df['prediction'] = np.where(df['money'] == '>50')
impute_attri()

X = np.array(df.drop(['money'], 1).values)
X = preprocessing.scale(X)
y = np.array(df['money'].values)

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size=0.2, random_state=101)
clf = SVC(degree=5)
# grid_param = {'C':[160,170,190], 'gamma': [0.0006,0.0007,0.0008]}
#
# grid = GridSearchCV(clf, grid_param, refit=True, verbose=4)
# grid.fit(X_train,y_train)
#
# with open('grid.pickle','wb') as f:
#     pickle.dump(grid, f)

pickle_in = open('grid.pickle', 'rb')
grid = pickle.load(pickle_in)


accuracy = grid.score(X_test,y_test)
print(grid.best_estimator_)
print(grid.best_params_)

print(grid.best_score_)
predict = grid.predict(X_test)
print(f'classification report:\n{classification_report(y_test,predict)}')
print(f'confusion matrix:\n{confusion_matrix(y_test,predict)}')
# print('MAE', metrics.mean_absolute_error(y_test, predict))
# print('MSE', metrics.mean_squared_error(y_test, predict))
# print('RMSE',np.sqrt(metrics.mean_squared_error(y_test, predict)))
# sb.heatmap(df.corr())
# sb.distplot((y_test-predict), bins=50)
plt.show()