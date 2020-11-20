import numpy as np

class MulGaussBC:
    '''
    Multivariate Gaussian Classifier

    This model is different from sklearn.naive_bayes.GaussianNB in that
    it will use the covariances between predictor variables.

    https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html
    https://github.com/scikit-learn/scikit-learn/blob/7e85a6d1f/sklearn/naive_bayes.py#L185
    https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    https://en.wikipedia.org/wiki/Naive_Bayes_classifier
    https://en.wikipedia.org/wiki/Matrix_normal_distribution
    '''

    def __init__(self, priors=None):
        self.means = []
        self.cov = []
        self.det_cov = []
        self.inv_cov = []
        self.classes = None
        self.priors = priors

    def fit(self, X, y):
        '''
        PARAMETERS
            X: Feature data (array-like).
            y: Training labels (array-like).
        '''
        self.classes = np.unique(y)
        if not self.priors:
            self.priors = np.ones(self.classes.shape) / self.classes.shape
        for k in self.classes:
            X_k = X[y == k]
            cov_k = np.cov(X_k.T)
            self.cov.append(cov_k)
            self.det_cov.append(np.linalg.det(cov_k))
            self.inv_cov.append(np.linalg.pinv(cov_k))
            self.means.append(np.mean(X_k, axis=0))

    def predict(self, x):
        '''
        Predict class of a single sample.
        '''
        predictions = []
        for k in range(len(self.classes)):
            prod = (2 * np.pi)**(-len(self.classes)/2)
            prod *= self.det_cov[k]**(-0.5)
            prod *= np.exp(-0.5*\
                           np.dot((x-self.means[k]).T,
                                  np.dot(self.inv_cov[k],
                                         (x-self.means[k]))))
            predictions.append(prod)
        predictions = np.array(predictions) * self.priors
        return np.argmax(predictions / np.sum(predictions))

class MatGaussBC:
    '''
    Matrix Normal Distribution Classifier

    https://en.wikipedia.org/wiki/Matrix_normal_distribution
    '''
    def __init__(self, priors=None):
        self.means = []
        self.cov_1 = []
        self.det_cov_1 = []
        self.inv_cov_1 = []
        self.cov_2 = []
        self.det_cov_2 = []
        self.inv_cov_2 = []
        self.classes = None
        self.priors = priors

    def fit(self, X, y):
        
        self.classes = np.unique(y)
        if not self.priors:
            self.priors = np.ones(self.classes.shape) / self.classes.shape
        for k in self.classes:
            print(k)
            X_k = X[y == k]
            self.means.append(np.mean(X_k, axis=0))
            cov_k_1 = np.mean(np.array([np.cov(i) for i in X_k]), axis=0)
            self.cov_1.append(cov_k_1)
            self.det_cov_1.append(np.linalg.det(cov_k_1))
            self.inv_cov_1.append(np.linalg.pinv(cov_k_1))
            cov_k_2 = np.mean(np.array([np.cov(i.T) for i in X_k]), axis=0)
            self.cov_2.append(cov_k_2)
            self.det_cov_2.append(np.linalg.det(cov_k_2))
            self.inv_cov_2.append(np.linalg.pinv(cov_k_2))

    def predict(self, x):
        predictions = []
        for k in range(len(self.classes)):
            prod = np.dot(self.inv_cov_1, (x-self.means[k]).T)
            prod = np.dot(prod, self.inv_cov_2)
            prod = np.dot(prod, (x-self.means[k]))
            prod = np.exp(-0.5 * np.trace(prod))
            prod /= (2*np.pi)**(self.means[k].shape[0]*self.means[k].shape[1]/2)
            prod /= self.det_cov_1[k] ** (self.means[k].shape[0]/2)
            prod /= self.det_cov_2[k] ** (self.means[k].shape[1]/2)
            predictions.append(prod)
        predictions = np.array(predictions)# * self.priors
        return np.argmax(predictions / np.sum(predictions))
                
            

if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn import model_selection
    dataset = load_iris()
    X = dataset['data']
    y = dataset['target']
    model = MulGaussBC()
    model.fit(X, y)
    p = np.array([model.predict(i) for i in X])
    print(np.sum(p == y) / len(p))
