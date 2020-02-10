import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.model_selection import KFold
from sklearn import tree
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression

df = pd. read_csv("DataSciencedataset.csv")
le = LabelEncoder()
df = df.rename(columns={"Unnamed: 27":"x27"})
df = df.rename(columns={"Unnamed: 28":"x28"})
df.drop(df.index[3952])
pd.value_counts(df['y'].values, sort=False)
df = df.fillna('0')

# label encoder for coulmns which are categorical
df.x1 = le.fit_transform(df.x1)
df.x2 = le.fit_transform(df.x2)
df.x3 = le.fit_transform(df.x3)
df.x4 = le.fit_transform(df.x4)
df.x5 = le.fit_transform(df.x5)
df.x8 = le.fit_transform(df.x8)
df.x15 = le.fit_transform(df.x15)
df.x16 = le.fit_transform(df.x16)
df.x17 = le.fit_transform(df.x17)
df.x23 = le.fit_transform(df.x23)
df.x24 = le.fit_transform(df.x24)
df.x25 = le.fit_transform(df.x25)
df.x26 = le.fit_transform(df.x26)
df.x27 = le.fit_transform(df.x27)


# the data is unbalanced and the number of the majarity of records are in 0 category of targets.
# to solve this problem we would make samples to increase the number of records with target 1.
df_majority = df[df.y == 0]
df_minority = df[df.y == 1]
df_minority_upsampled = resample(df_minority,replace=True,n_samples=9000,random_state=123)
df_upsampled = pd.concat([df_majority, df_minority_upsampled])
df_upsampled.y.value_counts()
df = df_upsampled


# selecting features that have a good corrolation with target
X = df[['x1',"x2",'x3','x4','x5','x6','x7','x8','x9',"x10",'x11','x12','x13',"x14","x15","x16",'x17','x18','x19',"x20",'x21','x23',"x24",'x25','x26','x27']]
y = df['y']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2)


# tree=tree.DecisionTreeClassifier()
# tree.fit(X_train,y_train)
# tree.score(X_test,y_test)


# svc = SVC()
# svc.fit(X_train,y_train)
# print(svc.score(X_test,y_test))
#


rfc = RandomForestClassifier()
rfc.fit(X_train,y_train)
print(rfc.score(X_test,y_test))



# lr=LogisticRegression()
# lr.fit(X_train,y_train)
# lr.score(X_test,y_test)
# print(lr.score(X_train,y_train))


# at first step confusion matrix show unbalance data in target
# so, resampling is required

y_predicted=rfc.predict(X_test)
cm= confusion_matrix(y_test,y_predicted)
print(cm)

# print(svc.predict(X_test[6:7]))
