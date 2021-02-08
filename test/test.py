import pandas as pd

from oneline import OneData

# input the dataset using simple OneData() with file location
data = OneData('./gender_submission.csv')

# print the OneData set
print(data)

# OneData is compatible with DataFrame
data_from_pd = pd.read_csv('gender_submission.csv')
data = OneData(data_from_pd)

# remove columns or rows using remove method
data = data.remove(column=['PassengerId'])

# summarize the data
data.summary()

# The function of DataFrame is also valid in OneData
data.to_csv('test_new.csv', index=False)
data = data.drop('PassengerId')  # drop() from DataFrame