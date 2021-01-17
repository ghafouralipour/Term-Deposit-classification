from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.dummy import DummyClassifier
from collections import Counter
from numpy import mean
from numpy import std
from numpy import hstack

# load the dataset
def load_dataset(full_path):
    # load the dataset as a numpy array
    dataframe = read_csv(full_path, header=None)
    # split into inputs and outputs
    last_ix = len(dataframe.columns) - 1
    X, y = dataframe.drop(last_ix, axis=1), dataframe[last_ix]
    target = dataframe.values[:, -1]
    counter = Counter(target)
    for k, v in counter.items():
        per = v / len(target) * 100
        print('Class=%s, Count=%d, Percentage=%.3f%%' % (k, v, per))
    # select categorical and numerical features
    cat_ix = X.select_dtypes(include=['object', 'bool']).columns
    num_ix = X.select_dtypes(include=['int64', 'float64']).columns
    # encode the target variable label to have the classes 0 and 1
    y = LabelEncoder().fit_transform(y)
    return X.values, y, cat_ix, num_ix

# evaluate the model
def evaluate_model(X, y, model):
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
    # evaluate model
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores

if __name__ == '__main__':
    # dataset path
    full_path = 'term-deposit-marketing-2020-2.csv'
    # load the dataset
    X, y, cat_ix, num_ix = load_dataset(full_path)
    # summarize the loaded dataset
    print(X.shape, y.shape, Counter(y))
    # define the reference model
    model = DummyClassifier(strategy='most_frequent')
    # evaluate the model
    scores = evaluate_model(X, y, model)
    # summarize performance
    print('Mean Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))


