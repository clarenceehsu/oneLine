from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier


class Supervised:
    def __init__(self):
        self.data = ''
        self.train_data = ''
        self.test_data = ''
        self.model = ''

    def linear_regression(self, x='', y=''):
        x_train = x
        y_train = y

        self.model = LinearRegression()
        self.model.fit(x_train, y_train)
        self.model.score(x_train, y_train)
        print('Coefficient: \n', self.model.coef_)
        print('Intercept: \n', self.model.intercept_)

    def logistic_regression(self, x='', y=''):
        x_train = x
        y_train = y

        self.model = LogisticRegression()
        self.model.fit(x_train, y_train)
        self.model.score(x_train, y_train)
        print('Coefficient: \n', self.model.coef_)
        print('Intercept: \n', self.model.intercept_)

    def decision_tree(self, x='', y=''):
        x_train = x
        y_train = y

        self.model = tree.DecisionTreeClassifier(criterion='gini')
        self.model.fit(x_train, y_train)
        self.model.score(x_train, y_train)

    def svm(self, x='', y=''):
        x_train = x
        y_train = y

        self.model = svm.svc()
        self.model.fit(x_train, y_train)
        self.model.score(x_train, y_train)

    def naive_bayes(self, x='', y=''):
        x_train = self.data[x]
        y_train = self.data[y]

        self.model = GaussianNB()
        self.model.fit(x_train, y_train)

    def kNN(self, x='', y='', n_nerghbors=6):
        x_train = x
        y_train = y

        self.model = KNeighborsClassifier(n_neighbors=n_nerghbors)
        self.model.fit(x_train, y_train)

    def kmeans(self, x='', n_clusters=6, random_state=0):
        x_train = x

        self.model = KMeans(n_clusters=n_clusters, random_state=random_state)
        self.model.fit(x_train)

    def random_forest(self, x='', y=''):
        x_train = x
        y_train = y

        self.model = RandomForestClassifier()
        self.model.fit(x_train, y_train)

    def predict(self, x_test):
        return self.model.predict(x_test)
