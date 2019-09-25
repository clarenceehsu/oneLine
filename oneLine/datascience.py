from .modules import *

def fast_plot(x, y, figsize=[], title='', xlabel='', ylabel='', smooth=False):
    sns.set()
    if figsize:
        plt.figure(figsize=figsize)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for (name, i) in y.items():
        if smooth:
            x_new = np.linspace(min(x), max(x), len(x) * 50)
            y_smooth = spline(x, i, x_new)
            plt.plot(x_new, y_smooth, label=name)
        else:
            plt.plot(x, i, label = name)
        plt.legend(loc='upper left')

    plt.show()

def comparing_variables(data,variable1, variable2):
    print(data[[variable1, variable2]][data[variable2].isnull()==False].groupby([variable1], as_index=False).mean().sort_values(by=variable2, ascending=False))
    g = sns.FacetGrid(data, col=variable2).map(sns.distplot, variable1)
def counting_values(data, variable1, variable2):
    return data[[variable1, variable2]][data[variable2].isnull()==False].groupby([variable1], as_index=False).mean().sort_values(by=variable2, ascending=False)
