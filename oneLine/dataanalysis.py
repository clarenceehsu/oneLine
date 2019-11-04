from .modules.data import *
from .modules.plot import *
from scipy import stats

def fill_na(data):
    for key, value in data.isnull().sum().items():
        if value:
            data[key].fillna(data[key].mode()[0], inplace=True)
    return data

def summary(df):
    pd.set_option('display.max_columns', None)
    print(f"Dataset Shape: {df.shape}")
    summ = pd.DataFrame(df.dtypes, columns=['dtypes'])
    summ = summ.reset_index()
    summ['Name'] = summ['index']
    summ = summ[['Name','dtypes']]
    summ['Missing'] = df.isnull().sum().values
    summ['Uniques'] = df.nunique().values
    summ['First Value'] = df.loc[0].values
    summ['Second Value'] = df.loc[1].values
    summ['Third Value'] = df.loc[2].values

    for name in summ['Name'].value_counts().index:
        summ.loc[summ['Name'] == name, 'Entropy'] = round(stats.entropy(df[name].value_counts(normalize=True), base=2),2)

    return summ

def comparing_variables(data,variable1, variable2):
    print(data[[variable1, variable2]][data[variable2].isnull() == False].groupby([variable1], as_index=False).mean().sort_values(by=variable2, ascending=False))
    sns.FacetGrid(data, col=variable2).map(sns.distplot, variable1)
    plt.show()

def counting_values(data, variable1, variable2):
    return data[[variable1, variable2]][data[variable2].isnull() == False].groupby([variable1], as_index=False).mean().sort_values(by=variable2, ascending=False)

def drop(data, column=[], index=[]):
    if column:
        data = data.drop(column, axis=1)
    if index:
        for n in index:
            data = data.drop(n)
    return data

