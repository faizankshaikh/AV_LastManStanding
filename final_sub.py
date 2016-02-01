# import modules
import pandas as pd
from xgboost import XGBClassifier as xgb
from sklearn.calibration import CalibratedClassifierCV as cccv
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

print 'Making Data'
train = pd.read_csv('Train.csv', index_col='ID')
test = pd.read_csv('Test.csv', index_col='ID')

#TODO
# try the golden attribute!

train.sort_values(by = 'Estimated_Insects_Count', inplace = True)
test.sort_values(by = 'Estimated_Insects_Count', inplace = True)

train.Number_Weeks_Used.fillna(method = 'ffill', inplace = True)
test.Number_Weeks_Used.fillna(method = 'ffill', inplace = True)

mms = MinMaxScaler()
X = mms.fit_transform(train.ix[:, train.columns != 'Crop_Damage'].values)
y = train.Crop_Damage.values

X_test = mms.transform(test.values)

X = X.astype('float32')
X_test = X_test.astype('float32')

train_x, test_x, train_y, test_y = train_test_split(X, y)

print 'Training Classifiers'
clf1 = xgb(nthread = 3, learning_rate = 0.3, n_estimators = 1000)
cccv1 = cccv(clf1, method='isotonic', cv = 5)

cccv1.fit(train_x, train_y);
pred1 = cccv1.predict(test_x)
print classification_report(test_y, pred1)

pred = cccv1.predict(X_test)
pd.DataFrame({'Crop_Damage':pred}, index=test.index).to_csv('final_sub.csv')
