#supressing library warnings 

import warnings
import numpy
warnings.filterwarnings('ignore')

import pandas #provides data structures to make data analysis easy


dataset = pandas.read_csv("train.csv")
dataset_test  = pandas.read_csv("test.csv")

ID = dataset_test['id']

#dropping the id column
dataset_test.drop('id', axis=1, inplace=True)

pandas.set_option('display.max_rows', None)
pandas.set_option('display.max_columns', None)


# print(dataset.head(1))
# print(dataset.shape)


print(dataset.skew())

# print(dataset.describe())

# Data prepation
#Machine Learning algorithms and python libraries to be used require numerical values

labels = []

split = 116
size = 15

data = dataset.iloc[:,split:]

cols = data.columns


for i in range(0, split):
    train = dataset[cols[i]].unique()
    test = dataset_test[cols[i]].unique()
    labels.append(list(set(train) | set(test)))



# del dataset_test

# Import OneHotEncoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

cats = []

for i in range(0, split):
    # label encode
    label_encoder = LabelEncoder()
    label_encoder.fit(labels[i])
    feature = label_encoder.transform(data.iloc[:,i])
    feature = feature.reshape(dataset_test.shape[0], 1)


    onehot_encoder = OneHotEncoder(sparse=False, n_values=len(labels[i]))
    feature = onehot_encoder.fit_transform(feature)
    cats.append(feature)

    # Make a 2D array from a list of 1D arrays
    encoded_cats = numpy.column_stack(cats)

    # Print encoded data's shape
    # print(encoded_cats.shape)
