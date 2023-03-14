from titanic.datasetup import *

# ---------------------------------- LOGIT MODEL ---------------------------------- # 0.77033
from sklearn.linear_model import LogisticRegression
from statsmodels.api import Logit
# To analyse the correlations in the datasets, we may apply OLS methods
# before ML methods. For this specific approach, we have to temporarily
# transform the data from categorical to numberical.

logit_model = LogisticRegression(max_iter = 500)
logit_model.fit(X_train, Y_train)

submission_logit = logit_model.predict(X_test)
submission_logit = pd.DataFrame(submission_logit, columns=['Survived'])
submission_logit.index = titanic_test.index
submission_logit.to_csv('titanic/submission.csv')

print(Logit(Y_train, X_train).fit().summary())


# ------------------------------ KNN Regressor ------------------------------ # 0.56459

from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=4, n_jobs=3)
X_train = titanic_train.drop(columns=['Survived','Ticket','Embarked'])
Y_train = titanic_train['Survived']

knn_model.fit(X_train, Y_train)

X_test = titanic_test.drop(columns=['Ticket','Embarked'])

submission_knn = knn_model.predict(X_test)
submission_knn = pd.DataFrame(submission_knn, columns=['Survived'])
submission_knn.index = titanic_test.index
submission_knn.to_csv('titanic/submission.csv')


# ------------------------------ Random Forest ------------------------------ # 0.75837

# One must create one-hot encoding the categorical variable in order to
# use it in the ML methods.

from sklearn.ensemble import RandomForestClassifier

random_forest_model = RandomForestClassifier()

random_forest_model.fit(X_train, Y_train)

submission_rf = random_forest_model.predict(X_test)
submission_rf = pd.DataFrame(submission_rf, columns=['Survived'])
submission_rf.index = titanic_test.index
submission_rf.to_csv('titanic/submission.csv')


# ------------------------------ XGBoost ------------------------------ # 0.75837

from xgboost import XGBClassifier
xgb_model = XGBClassifier(n_estimator = 17)

xgb_model.fit(X_train,Y_train)

submission_xgb = xgb_model.predict(X_test)
submission_xgb = pd.DataFrame(submission_xgb, columns=['Survived'])
submission_xgb.index = titanic_test.index
submission_xgb.to_csv('titanic/submission.csv')