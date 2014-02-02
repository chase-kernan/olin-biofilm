
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from classify import flat

def parse_data(path):
    samples = []
    with open(path, 'r') as f:
        for line in f:
            sample = [float(x) for x in line.split('\t') if x.strip()]
            if all(not np.isnan(x) for x in sample):
                samples.append(sample)
    return np.array(samples)

# classes
AEROBIC = 0
ANAEROBIC = 1
FLAT = 2

def get_data_and_classes():
    aerobicData = parse_data('classify/aerobic.txt')
    anaerobicData = parse_data('classify/anaerobic.txt')
    flatData = flat.generate(aerobicData.shape[1], numSamples=100)

    data = np.vstack((aerobicData, anaerobicData, flatData))
    classes = np.hstack((AEROBIC*np.ones(aerobicData.shape[0]),
                         ANAEROBIC*np.ones(anaerobicData.shape[0]),
                         FLAT*np.ones(flatData.shape[0])))
    return data, classes


def test_classifiers(n=20):
    data, classes = get_data_and_classes()
    #data = StandardScaler().fit_transform(data)

    def test(classifier):
        data_train, data_test, classes_train, classes_test = \
            train_test_split(data, classes, test_size=.4)
        classifier.fit(data_train, classes_train)
        return classifier.score(data_test, classes_test)

    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.naive_bayes import GaussianNB
    from sklearn.lda import LDA
    from sklearn.qda import QDA
    from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

    classifiers = {'SVC': SVC(gamma=2, C=1), 
                   'SVC Prob': SVC(gamma=2, C=1, probability=True),
                   'K-Neighbors': KNeighborsClassifier(),
                   'Linear SVC': SVC(kernel='linear', C=0.025),
                   'Linear SVC prob': SVC(kernel='linear', C=0.025, 
                                          probability=True),
                   'Random Forest': RandomForestClassifier(),
                   'Ada Boost': AdaBoostClassifier(),
                   'LDA': LDA(),
                   'QDA': QDA(),
                   'Gaussian NB': GaussianNB()}
    for name, classifier in classifiers.iteritems():
        score = np.median([test(classifier) for _ in range(n)])
        print name, score

    print classifiers['LDA'].predict_proba(parse_data('classify/aerobic.txt'))

def train():
    data, classes = get_data_and_classes()
    data = StandardScaler().fit_transform(data)

test_classifiers()
