from sklearn import svm
from sklearn import tree
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures


class MachineLearning(object):
    """
    That's the machine learning module for analysis using, and it's based on the scikit-learn module, which contains
    10 necessary ML methods.

    For example:
    a = MachineLearning(x=x, y=y)  # init with x, y data
    a.linear_reqression()  # train with x, y data
    a.predict(x_test)  # predict with the x_test data using trained model above
    """
    def __init__(self, x: list = None, y: list = None):
        """
        Initialze the dataset for modeling
        :param x: the x data
        :param y: the y data
        """
        self.x = x
        self.y = y
        self.model = None

    def set_value(self, x: list = None, y: list = None):
        """
        Reset value method
        :param x: the new x data
        :param y: the new y data
        """
        self.x = x
        self.y = y

    def linear_regression(self, degree: int = 1, include_bias: bool = False, info: bool = False):
        """
        Linear Regression Method
        :param degree: the degree
        :param include_bias: include bias or not
        :param info: show the info and parameters of the model
        """
        x_train = self.x
        y_train = self.y

        self.model = LinearRegression()
        if degree > 1:
            pf = PolynomialFeatures(degree=degree, include_bias=include_bias)
            x_train = pf.fit_transform(x_train)
        self.model.fit(x_train, y_train)
        self.model.score(x_train, y_train)
        if info:
            print('Coefficient: \n', self.model.coef_)
            print('Intercept: \n', self.model.intercept_)

    def logistic_regression(self, info: bool = False):
        """
        Logistic Regression Method
        :param info: show the info and parameters of the model
        """
        x_train = self.x
        y_train = self.y

        self.model = LogisticRegression()
        self.model.fit(x_train, y_train)
        self.model.score(x_train, y_train)
        if info:
            print('Coefficient: \n', self.model.coef_)
            print('Intercept: \n', self.model.intercept_)

    def decision_tree(self):
        """
        Decision Tree Method
        """
        x_train = self.x
        y_train = self.y

        self.model = tree.DecisionTreeClassifier(criterion='gini')
        self.model.fit(x_train, y_train)
        self.model.score(x_train, y_train)

    def svm(self):
        """
        Support Vector Machine Method
        """
        x_train = self.x
        y_train = self.y

        self.model = svm.svc()
        self.model.fit(x_train, y_train)
        self.model.score(x_train, y_train)

    def naive_bayes(self):
        """
        Naive Bayes Method
        """
        x_train = self.x
        y_train = self.y

        self.model = GaussianNB()
        self.model.fit(x_train, y_train)

    def knn(self, n_neighbors=6):
        """
        KNN methods
        :param n_neighbors: the number of n-neighbors
        """
        x_train = self.x
        y_train = self.y

        self.model = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.model.fit(x_train, y_train)

    def kmeans(self, n_clusters=6, random_state=0):
        """
        K-Means Method
        :param n_clusters: the number of clusters
        :param random_state: the number of random state
        """
        x_train = self.x

        self.model = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.model.fit(x_train)

    def random_forest(self):
        """
        Random Forest Method
        """
        x_train = self.x
        y_train = self.y

        self.model = RandomForestClassifier()
        self.model.fit(x_train, y_train)

    def predict(self, x_test: list):
        """
        Predict method for prediction
        :param x_test: the data for prediction
        """
        return self.model.predict(x_test)
