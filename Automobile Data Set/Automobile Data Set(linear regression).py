import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sb


pd.set_option('display.max_columns', 20)
df = pd.read_csv('Automobile Data Set.txt', delimiter=',', na_values=['?', -99999])
print(df.shape)
df.drop(['drive-wheels','num_of_cylinders','symboling','body-style','make'], 1, inplace=True)

def data_impute():
    df['num-of-doors'] = np.where(df['num-of-doors'] == 'two', 2, 4)
    df['engine_location'] = np.where(df['engine_location'] == 'front', 0, 1)
    # df['drive-wheels'] = np.where(df['drive-wheels'] == 'fwd', 0, 1)
    noc = {'three': 3,
           'two': 2,
           'four': 4,
           'five': 5,
           'six': 6,
           'seven': 7,
           'eight': 8,
           'nine': 9,
           'ten': 10,
           'eleven': 11,
           'twelve': 12}
    engine_type = {'dohc' : 100, 'ohcv': 102, 'ohc' : 101, 'l': 103,'rotor':104,'ohcf':105,'dohcv':106}
    fuel_tp = {'gas': 0, 'diesel': 1}
    fuel_sys = {'mfi': 15, 'mpfi': 10, '2bbl': 20, '1bbl': 30, 'spfi': 25, '4bbl':35, 'idi':40, 'spdi':45}
    aspir = {'std': 5, 'turbo': 10}
    body_style = {'convertible':11, 'hatchback':22,'sedan':33,'wagon':44,'hardtop':55}
    # df.num_of_cylinders = [noc[item] for item in df.num_of_cylinders]  # to iterate through rows and change the values
    df.fuel_type = [fuel_tp[x] for x in df.fuel_type]
    df.fuel_system = [fuel_sys[x] for x in df.fuel_system]
    df.aspiration = [aspir[y] for y in df.aspiration]
    df['engine-type'] = [engine_type[x] for x in df['engine-type']]
    # df['body-style'] = [body_style[x] for x in df['body-style']]
    # print(df['engine-type'])
    price_mean = df['price'].mean()
    norm_mean = df['normalized-losses'].mean()
    df['normalized-losses'] = df['normalized-losses'].fillna(norm_mean)
    horse_mean = df['horsepower'].mean()
    peak_rpm_mean = df['peak-rpm'].mean()
    bore_mean = df['bore'].mean()
    stroke_mean = df['stroke'].mean()
    # print(horse_mean)
    df['horsepower'] = df['horsepower'].fillna(horse_mean)
    df['peak-rpm'] = df['peak-rpm'].fillna(peak_rpm_mean)
    df['bore'] = df['bore'].fillna(bore_mean)
    df['stroke'] = df['stroke'].fillna(stroke_mean)
    df['price'] = df['price'].fillna(price_mean)
    # print(df.describe())


data_impute()
X = np.array(df.drop(['price'],1))
X = preprocessing.scale(X, 1)
y = np.array(df['price'])
clf = LinearRegression()

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=50)

clf.fit(X_train, y_train)
prediction = clf.predict(X_test)

print('MAE', metrics.mean_absolute_error(y_test, prediction))
print('MSE', metrics.mean_squared_error(y_test, prediction))
print('RMSE',np.sqrt(metrics.mean_squared_error(y_test, prediction)))
accuracy = clf.score(X_train, y_train)
print(f'accuracy: {accuracy}')

sb.distplot((y_test-prediction), bins=50)
# sb.heatmap(df.corr())
plt.show()