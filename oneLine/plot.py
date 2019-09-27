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

# def count_plot(dict, )