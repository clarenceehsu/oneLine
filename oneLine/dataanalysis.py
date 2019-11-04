from .modules.plot import *

def comparing_variables(data,variable1, variable2):
    print(data[[variable1, variable2]][data[variable2].isnull() == False].groupby([variable1], as_index=False).mean().sort_values(by=variable2, ascending=False))
    sns.FacetGrid(data, col=variable2).map(sns.distplot, variable1)
    plt.show()

def counting_values(data, variable1, variable2):
    return data[[variable1, variable2]][data[variable2].isnull() == False].groupby([variable1], as_index=False).mean().sort_values(by=variable2, ascending=False)