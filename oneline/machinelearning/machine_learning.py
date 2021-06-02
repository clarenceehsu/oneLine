"""
A wrapper for classical machine learning based on sci-kit learn. Because this simple wrapper is not
necessary, so this module would be deleted in the future.
"""
from ..tools.compat import import_optional_dependency


class MachineLearningModel(object):
    def __init__(self, model):
        self.model = model

    def fit_model(self, x, y, info):
        self.model.fit(x, y)
        self.model.score(x, y)
        if info:
            print('Coefficient: \n', self.model.coef_)
            print('Intercept: \n', self.model.intercept_)

    def predict(self, x_test: list):
        """
        Predict method for prediction
        :param x_test: the data for prediction
        """
        return self.model.predict(x_test)


def linear_regression(x, y, degree: int = 1, include_bias: bool = False, info: bool = False):
    """
    Linear Regression Method
    :param x: the x
    :param y: the y
    :param degree: the degree
    :param include_bias: include bias or not
    :param info: show the info and parameters of the model
    :return: model
    """
    # set the degree
    if degree > 1:
        pf = import_optional_dependency("sklearn.preprocessing")\
            .PolynomialFeatures(degree=degree, include_bias=include_bias)
        x = pf.fit_transform(x)

    # fit the model
    model = MachineLearningModel(import_optional_dependency("sklearn.linear_model").LinearRegression())
    model.fit_model(x, y, info)
    return model


def logistic_regression(x, y, info: bool = False):
    """
    Logistic Regression Method
    :param x: the x
    :param y: the y
    :param info: show the info and parameters of the model
    :return: model
    """
    # fit the model
    model = MachineLearningModel(import_optional_dependency("sklearn.linear_model").LogisticRegression())
    model.fit_model(x, y, info)
    return model


def decision_tree(x, y, criterion: str = "gini"):
    """
    Decision Tree Method
    :param x: the x
    :param y: the y
    :param criterion: the criterion method
    :return: model
    """
    # fit the model
    model = MachineLearningModel(
        import_optional_dependency("sklearn.tree").DecisionTreeClassifier(criterion=criterion))
    model.fit_model(x, y, info=False)
    return model


def svm(x, y, C: float = 1.0, kernel: str = 'rbf'):
    """
    Support Vector Machine Method
    :param x: the x
    :param y: the y
    :param C: the Regularization parameter
    :param kernel: Specifies the kernel type to be used in the algorithm
    :return: model
    """
    # fit the model
    model = MachineLearningModel(import_optional_dependency("sklearn.svm").SVC(C=C, kernel=kernel))
    model.fit_model(x, y, info=False)
    return model


def naive_bayes(x, y):
    """
    Naive Bayes Method
    :param x: the x
    :param y: the y
    :return: model
    """
    # fit the model
    model = MachineLearningModel(import_optional_dependency("sklearn.naive_bayes").GaussianNB())
    model.model.fit(x, y)
    return model


def knn(x, y, n_neighbors=6):
    """
    KNN methods
    :param x: the x
    :param y: the y
    :param n_neighbors: the number of n-neighbors
    :return: model
    """
    # fit the model
    model = MachineLearningModel(
        import_optional_dependency("sklearn.neighbors").KNeighborsClassifier(n_neighbors=n_neighbors))
    model.model.fit(x, y)
    return model


def kmeans(x, n_clusters=6, random_state=0):
    """
    K-Means Method
    :param x: the x
    :param n_clusters: the number of clusters
    :param random_state: the number of random state
    :return: model
    """
    # fit the model
    model = MachineLearningModel(
        import_optional_dependency("sklearn.cluster").KMeans(n_clusters=n_clusters, random_state=random_state))
    model.model.fit(x)
    return model


def random_forest(x, y):
    """
    Random Forest Method
    :param x: the x
    :param y: the y
    :return: model
    """
    model = MachineLearningModel(import_optional_dependency("sklearn.ensemble").RandomForestClassifier())
    model.model.fit(x, y)
    return model
