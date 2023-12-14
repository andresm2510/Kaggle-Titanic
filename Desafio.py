from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

test = pd.read_csv("CSV'S/test.csv")
train = pd.read_csv("CSV'S/train.csv")
test_ids = test["PassengerId"]
train.head()

df1 = train.drop(['Name', "PassengerId", 'Ticket', 'Cabin'], axis=1)
testData = test.drop(['Name', "PassengerId", 'Ticket', 'Cabin'], axis=1)
testData.head()

# Convertendo strings para valores númericos
df1['Sex'] = df1['Sex'].map({'female': 0, 'male': 1})
df1['Embarked'] = df1['Embarked'].map({'S': 0, 'C': 1, 'Q': 2, 'nan': 'NaN'})
testData['Sex'] = testData['Sex'].map({'female': 0, 'male': 1})
testData['Embarked'] = testData['Embarked'].map(
    {'S': 0, 'C': 1, 'Q': 2, 'nan': 'NaN'})

df1.isnull().sum()

testData.isnull().sum()

# Variaveis para armazenar a idade média de cada sexo
idadeMediaHomem0 = df1[df1['Sex'] == 1]['Age'].median()
idadeMediaMulher0 = df1[df1['Sex'] == 0]['Age'].median()
# Atribui o valor de idade média aos passageiros que possuem a idade como nula
df1.loc[(df1.Age.isnull()) & (df1['Sex'] == 1), 'Age'] = idadeMediaHomem0
df1.loc[(df1.Age.isnull()) & (df1['Sex'] == 0), 'Age'] = idadeMediaMulher0
idadeMediaHomem1 = testData[testData['Sex'] == 1]['Age'].median()
idadeMediaMulher1 = testData[testData['Sex'] == 0]['Age'].median()
testData.loc[(testData.Age.isnull()) & (
    testData['Sex'] == 1), 'Age'] = idadeMediaHomem1
testData.loc[(testData.Age.isnull()) & (
    testData['Sex'] == 0), 'Age'] = idadeMediaMulher1

df1.isnull().sum()

# verificando quais são os dois passassageiros com embarked nulo
EmbarkedNulo = df1[df1.Embarked.isnull()]
EmbarkedNulo

# Atribuindo um valor
df1 = df1.fillna(2)

# Preenchendo valores nulos na tarifa com a mediana
tarifaMedia = testData['Fare'].median()
testData.loc[testData.Fare.isnull(), 'Fare'] = tarifaMedia

df1.isnull().sum()


Y = df1["Survived"]
X = df1.drop("Survived", axis=1)

X_train, X_val, y_train, Y_val = train_test_split(
    X, Y, test_size=0.2, random_state=42)

clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, y_train)
predictions = clf.predict(X_val)
a = accuracy_score(Y_val, predictions)
print(a*100)

submission_preds = clf.predict(testData)

df = pd.DataFrame({"PassengerId": test_ids,
                  "Survived": submission_preds})
df.to_csv("submission.csv", index=False)
