
import numpy as np
import pandas as pd
import os
import nltk, openai, pandas, re, string
import sys

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


def create_master():

    pd.options.mode.chained_assignment = None 

    val = pd.read_csv("results/val.tsv", sep="\t", index_col=0)

    id_list = []
    id_list.extend(val['pair_id'].tolist())
    ids = pd.DataFrame(id_list, columns=['pair_id'])

    lab = []
    lab.extend(val['label'].tolist())
    lab = [1 if i == 'Y' else 0 for i in lab]
    labels = pd.DataFrame(lab, columns=['label'])

    master_df = pd.DataFrame()
    master_df = pd.concat([master_df, ids, labels], axis=1, ignore_index=False)


    sets = ["H30", "R01", "R02", "R03"]
    len_dict ={'H30':70, 'R01': 111, 'R02':81, 'R03':109} 



    dir_path_hug = './results/huggingface/'

    prefixed_hug = [filename for filename in os.listdir(dir_path_hug) if "H30" in os.path.splitext(filename)[0]]

    for file in prefixed_hug:
        col_name = file.replace("-H30", "")
        answers = []
        for name in sets: 
            file2 = file.replace("H30", name)
            try:
                read_file = pd.read_csv(dir_path_hug + file2, sep="\t", index_col=0)
                ans = read_file['model_predictions']
            except:
                num = len_dict[name]
                ans = ['NaN']*num 
            answers.extend(ans)

        ans_list = []
        for i in range(len(answers)):
            if answers[i]==1:
                ans_list.append(1)
            elif answers[i]==0:
                ans_list.append(0)
            else:
                ans_list.append(2)



        column = pd.DataFrame(ans_list, columns=[col_name])    
        master_df = pd.concat([master_df, column], axis=1, ignore_index=False)

    return val, master_df


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

    train = pd.read_csv("results/train.tsv", sep="\t", index_col=0)
    # Preprocess the input strings and data strings
    preprocessed_train_list = [preprocess(row["articles"] + "\n\n" + row["query"]) for i, row in train.iterrows()]
    preprocessed_val_list = [preprocess(row["articles"] + "\n\n" + row["query"]) for i, row in data.iterrows()]
    # Create a TF-IDF vectorizer and fit it to the preprocessed input and data strings
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_train_list)
    total_features = len(vectorizer.vocabulary_)
    vector = TfidfVectorizer(max_features=int(total_features * 0.1))
    # tfidf_matrix_tr = vector.fit_transform(preprocessed_train_list)
    tfidf_matrix = vector.fit_transform(preprocessed_val_list)


    tf_arr = tfidf_matrix.toarray()
    tf_pd = pd.DataFrame(tf_arr, columns=[str(i) for i in range(tf_arr.shape[1])])

    concat_df = pd.concat([dataframe,tf_pd], axis=1)

    return concat_df


def get_sample_acc(dataframe):
    all_accs = []
    for i in range(dataframe.shape[0]):
        samp_pred = dataframe.iloc[i,2:]
        count = (dataframe['label'][i]==samp_pred).astype(int)
        samp_acc = sum(count)/len(dataframe.columns.tolist()[2:])
        all_accs.append(samp_acc)
    acc_df = pd.DataFrame(all_accs, columns=['samp_acc'])
    
    pd.set_option('display.max_rows', None)

    

    font = {'family' : 'normal', 'weight' : 'normal', 'size'   : 11}
    mpl.rc('font', **font)
    mpl.use('TkAgg')
    n_bins = 10
    sns.set_style("dark")
    fig, axs = plt.subplots(1, 1, sharey=True, tight_layout=True)
    axs.hist(np.array(acc_df), bins=n_bins)
    axs.set(xlabel='Accuracy score', ylabel='Number of Predictions')
    axs.grid()
    # plt.show()

    dataframe = pd.concat([dataframe, acc_df], axis=1, ignore_index=False)
    return  np.array(acc_df) # dataframe


def calc_acc(algorithm = 'svm'):

    val, master_df = create_master()
    hist_arr = get_sample_acc(master_df)

    h = []
    for i in hist_arr:
        if i >=0.5:
            h.append(1)
        elif i<0.5:
            h.append(0)
        else:
            h.append(2)

    num_models = len(master_df.columns.tolist()[2:])

    master_df = get_tfidf_feat(val, master_df)

    feature_cols = master_df.columns.tolist()[2:]

    x = master_df[feature_cols] # Features
    y = master_df.label # Target variable

    if algorithm == 'rf': 
        print('Random Forest:')  
        clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=0)
    elif algorithm == 'svm':
        print('Support Vector Machines:')  
        clf = svm.SVC()


    scores = cross_val_score(clf, x, h, cv=5)
    print("Mean Accuracy {}-Fold: ".format(len(scores)), sum(scores)/len(scores))



def main():
    """
    To run 'Support Vector Machines' enter:
    python dt.py svm
    To run 'Random Forest' enter:
    python dt.py rf
    """

    if len(sys.argv) >= 2:
        if sys.argv[1] == 'rf':
            alg = 'rf'
        else:
            alg = 'svm'    
    else:
        alg = 'svm'
    calc_acc(alg)


if __name__ == "__main__":
    main () 






