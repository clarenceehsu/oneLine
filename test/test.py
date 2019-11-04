from ..oneLine.data import OneData
from ..oneLine.data import OneDatalist

path = r'gender_submission.csv'
data = OneData(filepath=path)

data = data.drop(column=['PassengerId'])
print(data.summary)
data.save_csv('test.csv')