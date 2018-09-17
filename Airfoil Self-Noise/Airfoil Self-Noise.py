import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import preprocessing, metrics
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sb
from sklearn.preprocessing import imputation, Imputer

pd.set_option('display.max_columns', 12)
df = pd.read_csv('Airfoil Self-Noise.txt', delimiter='\t')
mean = df['frequency(hz)'].mean()

# df['frequency(hz)'] = np.where(df['frequency(hz)'] > 7000, mean,  df['frequency(hz)'])
# sb.pairplot(df)
# plt.show()
print(df.describe())
X = np.array(df.drop(["scaled_sound_pressure"], 1).values)
encoder = preprocessing.LabelEncoder()
y = np.array(df["scaled_sound_pressure"].values)
# train_score = encoder.fit_transform(y)
# print(train_score)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)
clf = LinearRegression()
clf.fit(X_train, y_train)

# coeff_df = pd.DataFrame(clf.coef_, X.columns, columns=['Coefficient'])
# print(coeff_df)
predict = clf.predict(X_test)
sb.distplot((y_test-predict), bins=50)
# plt.show()
print('MAE:', metrics.mean_absolute_error(y_test, predict))
print('MSE:', metrics.mean_squared_error(y_test, predict))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predict)))
accuracy = clf.score(X_test, y_test)
print(f'accuracy:{accuracy}')
# confusion = confusion_matrix((X_test, y_test))
# print(confusion)