import pandas as pd

from ..oneLine import OneData
from ..oneLine import test_enviorment

# test if there are modules not installed
test_enviorment()

# input the dataset using simple OneData()
data = OneData('gender_submission.csv')

# print the OneData set
print(data)

# OneData is compatible with DataFrame
data_from_pd = pd.read_csv('gender_submission.csv')
data = OneData(data_from_pd)

# remove columns or rows using remove method
data = data.remove(column=['PassengerId'])

# summerize the data
data.summary()

# OneData can use the methods of DataFrame
a.to_csv('test_new.csv', index=False)