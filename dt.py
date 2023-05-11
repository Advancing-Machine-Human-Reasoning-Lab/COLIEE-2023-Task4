import numpy as np
import pandas as pd
import os
import torch
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.datasets import make_classification

from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import random

import nltk, openai, pandas, random, re, string

# from apikeys import open_ai, org_id
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


# for i in range(10):
# print(random.randint(0,1))

pd.options.mode.chained_assignment = None

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


val = pd.read_csv("COLIEE2023statute_data-English/val.tsv", sep="\t", index_col=0)

id_list = []
id_list.extend(val["pair_id"].tolist())
ids = pd.DataFrame(id_list, columns=["pair_id"])

lab = []
lab.extend(val["label"].tolist())
lab = [1 if i == "Y" else 0 for i in lab]
labels = pd.DataFrame(lab, columns=["label"])

master_df = pd.DataFrame()
master_df = pd.concat([master_df, ids, labels], axis=1, ignore_index=False)


sets = ["H30", "R01", "R02", "R03"]
len_dict = {"H30": 70, "R01": 111, "R02": 81, "R03": 109}


dir_path_hug = "./results/huggingface/"
# prefixed_hug = [filename for filename in os.listdir(dir_path_hug) if os.path.splitext(filename)[0].endswith("H30")]

prefixed_hug = [
    filename
    for filename in os.listdir(dir_path_hug)
    if "H30" in os.path.splitext(filename)[0]
]

for file in prefixed_hug:
    col_name = file.replace("-H30", "")
    answers = []
    for name in sets:
        file2 = file.replace("H30", name)
        try:
            read_file = pd.read_csv(dir_path_hug + file2, sep="\t", index_col=0)
            ans = read_file["model_predictions"]
        except:
            num = len_dict[name]
            ans = ["NaN"] * num
        answers.extend(ans)

    ans_list = []
    for i in range(len(answers)):
        if answers[i] == 1:
            ans_list.append(1)
        elif answers[i] == 0:
            ans_list.append(0)
        else:
            # print(name)
            # print(col_name)
            # ans_list.append(random.randint(0,1))
            ans_list.append(2)

    column = pd.DataFrame(ans_list, columns=[col_name])
    master_df = pd.concat([master_df, column], axis=1, ignore_index=False)


# Define a function to preprocess text
def preprocess(text):
    # Lowercase the text
    text = text.lower()
    # Remove punctuation and unwanted characters
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Tokenize the text into words
    words = nltk.word_tokenize(text)
    # Remove stop words
    words = [word for word in words if word not in stopwords.words("english")]
    # Lemmatize the words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    # Join the words back into a string
    text = " ".join(words)
    return text


def get_tfidf_feat(data, dataframe):
    train = pd.read_csv(
        "COLIEE2023statute_data-English/train.tsv", sep="\t", index_col=0
    )
    # Preprocess the input strings and data strings
    preprocessed_train_list = [
        preprocess(row["articles"] + "\n\n" + row["query"])
        for i, row in train.iterrows()
    ]
    preprocessed_val_list = [
        preprocess(row["articles"] + "\n\n" + row["query"])
        for i, row in data.iterrows()
    ]
    # Create a TF-IDF vectorizer and fit it to the preprocessed input and data strings
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_train_list)
    total_features = len(vectorizer.vocabulary_)
    vector = TfidfVectorizer(max_features=int(total_features * 0.001))
    # tfidf_matrix_tr = vector.fit_transform(preprocessed_train_list)
    tfidf_matrix = vector.fit_transform(preprocessed_val_list)

    tf_arr = tfidf_matrix.toarray()
    tf_pd = pd.DataFrame(tf_arr, columns=[str(i) for i in range(tf_arr.shape[1])])

    concat_df = pd.concat([dataframe, tf_pd], axis=1)

    return concat_df


# The uncertain rows:
# u_list = [24, 37, 41, 45, 55, 77, 85, 86, 87, 89, 90, 100, 101, 102, 109, 123, 127, 128, 130, 133, 134, 143, 144, 146,
# 148, 149, 154, 157, 158, 159, 161, 169, 174, 175, 182, 183, 197, 210, 229, 247, 249, 262, 297, 301, 302, 303, 311, 313,
# 319, 327, 328, 342, 364]


print(len(master_df.columns.tolist()[2:]))


def get_sample_acc(dataframe):
    all_accs = []
    for i in range(dataframe.shape[0]):
        samp_pred = dataframe.iloc[i, 2:]
        count = (dataframe["label"][i] == samp_pred).astype(int)
        samp_acc = sum(count) / len(dataframe.columns.tolist()[2:])
        all_accs.append(samp_acc)
    acc_df = pd.DataFrame(all_accs, columns=["samp_acc"])

    # pd.set_option('display.max_rows', None)
    # print(acc_df)
    # print(np.histogram(np.array(acc_df)))

    # import matplotlib.pyplot as plt
    # n_bins = 10
    # fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
    # axs.hist(np.array(acc_df), bins=n_bins)

    # plt.show()
    # input()
    dataframe = pd.concat([dataframe, acc_df], axis=1, ignore_index=False)
    return dataframe  # np.array(acc_df) # dataframe


# hist_arr = get_sample_acc(master_df)
master_df = get_sample_acc(master_df)

print(master_df)
input()

master_df.to_csv("master_df.tsv", sep="\t", encoding="utf-8")
input()
# print(hist_arr)

h = []
for i in hist_arr:
    if i > 0.7:
        h.append(1)
    elif i < 0.3:
        h.append(0)
    else:
        h.append(2)


# h = [1 if i>0.7 else 0 for i in hist_arr]
# print(hist_arr)

# input()


num_models = len(master_df.columns.tolist()[2:])
tf_start = num_models + 2

master_df = get_tfidf_feat(val, master_df)
# print(master_df)
# input()
# df_range = list(np.arange(371))
# diff = list(set(df_range) - set(u_list))

# the rows in u_list are test samples. The others are train samples.
# select_df = master_df.loc[u_list]
# diff_df = master_df.loc[diff]

# If feature_cols=master_df.columns.tolist()[2:] it considers the tfidf features.
feature_cols = master_df.columns.tolist()[2:28]
# tfidf_start = num_models+2
# tf_cols = master_df.columns.tolist()[tfidf_start:]
# feature_cols = ['flan-t5-xxl-tfidf-balanced-pruned-0.15-2shots-1234.tsv', 'flan-t5-xxl-tfidf-balanced-2shots-1234.tsv', 'flan-ul2-vanilla-1234.tsv',
# 'T0pp-vanilla-1234.tsv', 'flan-t5-xxl-vanilla-1234.tsv', 'flan-t5-xxl-tfidf-balanced-pruned-0.9-2shots-1234.tsv',
# 'flan-t5-xxl-tfidf-balanced-4shots-1234.tsv', 'flan-t5-xxl-tfidf-balanced-pruned-0.75-2shots-1234.tsv', 'flan-t5-xxl-tfidf-balanced-pruned-0.3-2shots-1234.tsv',
# 'flan-t5-xxl-tfidf-3shots-9999.tsv', 'T0p-vanilla-1234.tsv', 'flan-t5-xxl-tfidf-unbalanced-3shot-1234.tsv',
# 'flan-t5-xxl-tfidf-balanced-pruned-0.6-2shots-1234.tsv', 'flan-t5-xxl-tfidf-3shots-1234.tsv'
# ]
# feature_cols = ['flan-t5-xxl-tfidf-balanced-4shots-1234.tsv', 'flan-t5-xxl-tfidf-balanced-pruned-0.75-2shots-1234.tsv', 'flan-t5-xxl-tfidf-balanced-pruned-0.3-2shots-1234.tsv']
# feature_cols = ['flan-ul2-vanilla-1234.tsv', 'T0pp-vanilla-1234.tsv', 'flan-t5-xxl-vanilla-1234.tsv', 'T0-vanilla-1234.tsv',
# 'flan-t5-xxl-tfidf-balanced-4shots-1234.tsv', 'flan-t5-xxl-tfidf-3shots-9999.tsv', 'T0_3B-vanilla-1234.tsv', 'T0p-vanilla-1234.tsv',
# 'flan-t5-xxl-tfidf-unbalanced-3shot-1234.tsv', 'flan-t5-xxl-tfidf-3shots-1234.tsv']
# feature_cols.extend(tf_cols)
print(feature_cols)

x = master_df[feature_cols]  # Features
y = master_df.label  # Target variable


# Print all columns it considers.

######
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import make_pipeline

# from tqdm import tqdm
from warnings import simplefilter
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest, chi2, RFE, mutual_info_classif


def get_pipeline(k, classifier):
    # if over_sampling:
    sampler = RandomOverSampler(random_state=42, sampling_strategy="minority")
    # else:
    # sampler = RandomUnderSampler(random_state=13, sampling_strategy  = 'majority')
    feature_selection = SelectKBest(chi2, k=k)
    # if normalize:
    # pipe_classifier = make_pipeline(Normalizer(),sampler,feature_selection,classifier)
    # else:
    pipe_classifier = make_pipeline(classifier)
    # pipe_classifier = make_pipeline(feature_selection,classifier)
    # pipe_classifier = make_pipeline(sampler,classifier)

    return pipe_classifier


from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

# ############################333

# print(feature_cols)
y2 = h

# clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
# clf = tree.DecisionTreeClassifier(max_depth=2)
clf = svm.SVC()
# clf = LogisticRegression()
# clf = AdaBoostClassifier(DecisionTreeClassifier(criterion = 'entropy',max_depth = 5),n_estimators = 100)
pipe_clf = get_pipeline(3, clf)

scores = cross_val_score(pipe_clf, x, y2, cv=5)

y_pred = cross_val_predict(pipe_clf, x, y2, cv=10)
conf_mat = confusion_matrix(y2, y_pred)
print(scores)
print("Mean Accuracy {}-Fold: ".format(len(scores)), sum(scores) / len(scores))
print(conf_mat)
# exit()

#############################

# clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
# clf = tree.DecisionTreeClassifier(max_depth=2)
clf = svm.SVC()
# clf = AdaBoostClassifier(DecisionTreeClassifier(criterion = 'entropy',max_depth = 5),n_estimators = 100)
pipe_clf = get_pipeline(3, clf)

scores = cross_val_score(pipe_clf, x, y, cv=5)

y_pred = cross_val_predict(pipe_clf, x, y, cv=10)
conf_mat = confusion_matrix(y, y_pred)
print(scores)
print("Mean Accuracy {}-Fold: ".format(len(scores)), sum(scores) / len(scores))
print(conf_mat)
# print("Mean Accuracy {}-Fold: ".format(len(scores)), (conf_mat[0,0]+conf_mat[1,1])/(sum(conf_mat[0,:])+sum(conf_mat[1,:])))


# print(confusion_matrix(y_test,y_pred))
######

# Feature Selection:
# feature_selection = SelectKBest(chi2,k=10)
# feature_selection.fit_transform(x,y)
# filterfeat = feature_selection.get_support()
# features = np.array(x.columns)
# print(features[filterfeat])


###################
