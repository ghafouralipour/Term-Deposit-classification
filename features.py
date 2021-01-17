from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor


def encode_dataset(input_full_path,output_full_path):
    # load the dataset as a numpy array
    headers=['age','job','marital','education','default','balance','housing','loan','contact','day','month','duration','campaign','y']
    dataframe = read_csv(input_full_path, header=None,names=headers, na_values="?")
    cleanup_nums = {"job":     {"blue-collar": 1, "management": 2,"technician":3,"admin":4,"services":5,
                                "retired":6,"self-employed":7,"entrepreneur":8,"unemployed":9,
                                "housemaid":10,"student":11,"unknown":12},
                    "marital":{"married": 1, "single": 2,"divorced":3},
                    "education":{"secondary": 1, "tertiary": 2,"primary":3,"unknown":4},
                    "default":{"no": 1, "yes": 2},
                    "housing":{"no": 1, "yes": 2},
                    "loan":{"no": 1, "yes": 2},
                    "contact":{"cellular": 1, "unknown": 2,"telephone":3},
                    "month":     {"may": 1, "jul": 2,"aug":3,"jun":4,"nov":5,
                                "apr":6,"feb":7,"jan":8,"mar":9,
                                "oct":10,"dec":11,"sep":12},
                    "y":{"no": 0, "yes": 1},
                   }
    dataframe = dataframe.replace(cleanup_nums)
    dataframe.to_csv(output_full_path, header=False, index=False)
    return dataframe


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
    # label encode the target variable to have the classes 0 and 1
    y = LabelEncoder().fit_transform(y)
    return X.values, y, cat_ix, num_ix

if __name__ == '__main__':
    # define the location of the dataset
    in_full_path = 'term-deposit-marketing-2020-2.csv'
    out_full_path = 'termDMPure.csv'

    dataframe = encode_dataset(in_full_path, out_full_path)
    X, y, cat_ix, num_ix = load_dataset(out_full_path)
    # define the model
    model = DecisionTreeRegressor()
    # fit the model
    model.fit(X, y)
    # get importance
    importance = model.feature_importances_
    # summarize feature importance
    for i, v in enumerate(importance):
        print('Feature: %0d, Score: %.5f' % (i, v))
    # plot feature importance
    plt.bar([x for x in range(len(importance))], importance)