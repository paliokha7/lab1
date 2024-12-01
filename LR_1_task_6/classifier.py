from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

def train_svm(x_train, y_train):
    svm_classifier = SVC(kernel='linear')
    svm_classifier.fit(x_train, y_train)
    return svm_classifier

def train_nb(x_train, y_train):
    nb_classifier = GaussianNB()
    nb_classifier.fit(x_train, y_train)
    return nb_classifier


