# Kaggle Competition - Titanic, Attempt - 1

import pandas as pd
import matplotlib.pyplot as plt
import ydata_profiling as ydp

titanic_train = pd.read_csv("titanic/train.csv")
titanic_test = pd.read_csv("titanic/test.csv")

#  The PassengerId and Name column must be index.
titanic_train.set_index(['PassengerId', 'Name'], inplace=True)
titanic_test.set_index('PassengerId', inplace=True)
titanic_test.drop(columns='Name', inplace=True)

# Create a Profile Report of the dataset before interpolation of missing
# data, so that it can be compared with the post-processed dataset.
#ydp.ProfileReport(titanic_train).to_file("titanic/BeforeImputation.html")

titanic_train.Sex.unique()
# The Sex should be transformed into binary/dummy variable,
# where male=1 and female=0.
titanic_train.Sex.replace(
    ['male', 'female'], [1, 0], inplace = True
)
titanic_test.Sex.replace(
    ['male', 'female'], [1, 0], inplace = True
)

# The original data doesn't shows duplication. Note that the
# transformed data shows that, but the index verifies that all
# of such observation belongs to a different individual.
titanic_train.index.is_unique

# For the missing values first we must count how many NaNs are in
# each column, and then decide whether to drop them.

print(
    'Usable Non-nan data per column \n',
    100*titanic_train.count()/titanic_train.shape[0]
)
print(
    'Usable Non-nan data per column \n',
    100*titanic_test.count()/titanic_test.shape[0]
)

# With 80 percent and 99 percent non-NaN data in Age and Embarked,
# it must be imputed in the dataset. For the Embarked, rather than
# dropping the column or data, one may make it a cateogory including
# NA as one of the category.

titanic_train['Cabin'].fillna('NA', inplace = True)
titanic_train['Cabin'] = [i if i == 'NA' else i[0] for i in titanic_train['Cabin']]
titanic_train['Cabin'] = titanic_train['Cabin'].astype('category')
titanic_train['Cabin'][titanic_train['Cabin'] == 'T']
titanic_train['Cabin'] = titanic_train['Cabin'].cat.remove_categories('T')

titanic_train['Embarked'].replace(
    ['C', 'Q', 'S'],
    ['Cherbourg', 'Queenstown', 'Southampton'],
    inplace=True
)
titanic_train['Embarked'] = titanic_train['Embarked'].astype('category')
titanic_train['Embarked'] = titanic_train['Embarked'].cat.add_categories('NA')
titanic_train['Embarked'].fillna('NA', inplace = True)

titanic_test['Cabin'].fillna('NA', inplace = True)
titanic_test['Cabin'] = [i if i == 'NA' else i[0] for i in titanic_test['Cabin']]
titanic_test['Cabin'] = titanic_test['Cabin'].astype('category')

titanic_test['Embarked'].replace(
    ['C', 'Q', 'S'],
    ['Cherbourg', 'Queenstown', 'Southampton'],
    inplace=True
)
titanic_test['Embarked'] = titanic_test['Embarked'].astype('category')
titanic_test['Embarked'] = titanic_test['Embarked'].cat.add_categories('NA')
titanic_test['Embarked'].fillna('NA', inplace = True)

# Still, dropping the missing data doesn't seems to intuitive,
# since dropping the NaNs would lead to a decrement of around 20%.
print(
    'UnUsable nan Data\n',
    100 - 100*titanic_train.dropna().count()[0]/titanic_train.shape[0],
    '%'
)
print(
    'UnUsable nan Data\n',
    100 - 100*titanic_test.dropna().count()[0]/titanic_test.shape[0],
    '%'
)

# Hence, interpolation should be attempted, such that the relationship
# betweenthe features doesn't change drastically. Also, the target
# variable Survived should be omitted during interpolation. kNN is
# advised by the internet for interpolation of missing data.

# plt.hist(titanic_train['Age'], color='grey', ec='black')
# plt.hist(titanic_test['Age'], color='grey', ec='black')
# plt.hist(titanic_train['Cabin'], color='grey', ec='black')
# plt.show()

# The kNN method for imputation can be executed. It can be seen that Age is
# 'almost' normally distributed, and a minimum of 25 people exists in 8
# groups. We can put number of neighbours to 8.

from sklearn.impute import KNNImputer

knn_train = KNNImputer(n_neighbors=8)
titanic_train['Age'] = knn_train.fit_transform(titanic_train[['Age']])
titanic_train['Embarked'].replace('NA', titanic_train['Embarked'].mode()[0], inplace=True)

knn_test = KNNImputer(n_neighbors=5)
titanic_test['Age'] = knn_test.fit_transform(titanic_test[['Age']])
titanic_test['Embarked'].replace('NA', titanic_test['Embarked'].mode()[0], inplace=True)

# As can be seen, the remaining missing data can be dropped without
# significant damage, since the number of missing data in test amounts
# now only for less than half a percent.

print('UnUsable nan Data\n' , 100 - 100*titanic_train.dropna().count()[0]/titanic_train.shape[0],'%')
print('UnUsable nan Data\n' , 100 - 100*titanic_test.dropna().count()[0]/titanic_test.shape[0],'%')

titanic_train.dropna(inplace=True)

# However, the titanic_test requires all the rows, and hence, the Fare
# needs to be imputed.

knn_test = KNNImputer(n_neighbors=2)
titanic_test['Fare'] = knn_test.fit_transform(titanic_test[['Fare']])

# Create a Profile Report of the dataset after fixing missing data, so that
# it can be compared with the post-processed dataset.
#ydp.ProfileReport(titanic_train).to_file("titanic/AfterImputation.html")

# The train and test data can be Dummyfied as below.

X_train = pd.concat([
    titanic_train.drop(columns=['Survived','Ticket','Embarked','Cabin']),
    pd.get_dummies(titanic_train['Embarked']),
    pd.get_dummies(titanic_train['Cabin']),
], axis=1)

Y_train = titanic_train['Survived']

X_test = pd.concat([
    titanic_test.drop(columns=['Ticket','Embarked','Cabin']),
    pd.get_dummies(titanic_test['Embarked']),
    pd.get_dummies(titanic_test['Cabin'])
], axis=1)