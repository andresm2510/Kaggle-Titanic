import pandas as pd

test = pd.read_csv("test.csv")
train= pd.read_csv("train.csv")
test_ids = test["PassengerId"]
train.head(5)

def clean(tenta):
    x = tenta.drop(["Ticket","Cabin", "Name", "PassengerId"], axis=1)
    colunas = ["SibSp", "Parch" , "Fare" , "Age"]

    for coluna in colunas:
        x[coluna].fillna(x[coluna].median(), inplace=True)

    x.Embarked.fillna("U", inplace=True)
    return x

train= clean(train)
test=clean(test)
train.head(5)

train['Sex'] = train['Sex'].map({'female':0, 'male':1})
train['Embarked'] = train['Embarked'].map({'S':0, 'C':1, 'Q':2, '':0})
train = train.fillna(0)
train.head(5)
test['Sex'] = test['Sex'].map({'female':0, 'male':1})
test['Embarked'] = test['Embarked'].map({'S':0, 'C':1, 'Q':2, '':0})
test = test.fillna(0)
a= train.isnull().sum()
print(a)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

Y = train["Survived"]
X = train.drop("Survived", axis=1)

X_train, X_val, y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

clf= LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)
predictions = clf.predict(X_val)
from sklearn.metrics import accuracy_score
a= accuracy_score(Y_val, predictions)
print(a)

submission_preds = clf.predict(test)

df= pd.DataFrame({"PassengerId": test_ids, 
                  "Survived": submission_preds})
df.to_csv("submission.csv", index=False)