import argparse, nltk, numpy, os, pandas, random, re, similarity, string

from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from termcolor import colored
from tqdm import tqdm


argparser = argparse.ArgumentParser()
argparser.add_argument(
    "--model_name",
    type=str,
    default="gpt-3.5-turbo",
    help="Model name. Can be any one of: 'gpt-3.5-turbo', 'gpt-4', or any huggingface models.",
)
argparser.add_argument(
    "--prompting_style",
    type=str,
    default="vanilla",
    help="Prompting style. One of 'vanilla', 'select-most-relevant', 'consider-both', 'self-ask', 'information-explanation'.",
)
argparser.add_argument(
    "--splits",
    type=str,
    default="H30,R01,R02,R03",
    help="Comma-separated list of splits to evaluate on. One of more of 'H30,R01,R02,R03,Test,train'.",
)
argparser.add_argument(
    "--num_shots", type=int, default=0, help="Number of shots to use. 0 means no shots."
)
argparser.add_argument(
    "--similarity_metric",
    default=None,
    help="Similarity metric to use. One of 'TFIDF', 'sbert', 'bleurt', 'bert-score', and 'all'. Only relevant when num_shots > 0.",
)
argparser.add_argument(
    "--temperature_sensitive_ensemble",
    default=False,
    action="store_true",
    help="Whether to use a temperature-sensitive ensemble. Automatically sets ensemble size to 5. Is automatically set to True when prompting style is 'information-explanation'.",
)
argparser.add_argument(
    "--ensemble_size",
    type=int,
    default=1,
    help="Number of models to ensemble. Needs to be > 1 when temperature-sensitive ensemble is used.",
)
argparser.add_argument(
    "--temperature",
    type=int,
    default=1,
    help="Temperature to use. Irrelevant when temperature-sensitive ensemble is used.",
)
argparser.add_argument(
    "--overwrite",
    action="store_true",
    default=False,
    help="Whether to overwrite existing results.",
)
argparser.add_argument(
    "--handleU",
    default="random",
    help="How to handle unknown labels. Only when One of prompting style is 'information-explanation'. One of 'random', 'legal_terminology', 'similar_articles', 'similar_explanation', 'similar_questions', 'similar_articles_explanation', 'similar_articles_questions', 'similar_explanation_questions', 'similar_articles_explanation', 'similar_articles_questions', 'similar_explanation_questions', 'similar_articles_explanation_questions'.",
)
argparser.add_argument(
    "--do_sample",
    action="store_true",
    default=False,
    help="Whether to sample from the model.",
)
argparser.add_argument(
    "--balanced",
    action="store_true",
    default=False,
    help="Whether to balanced the shots.",
)
argparser.add_argument(
    "--similar_shots",
    default="yes",
    help="Whether to use similar shots. Only relevant when num_shots > 0. One of 'yes', 'no', 'mixed'.",
)
argparser.add_argument(
    "--separate_shots",
    action="store_true",
    default=False,
    help="Whether an ensemble has separate shots. Only relevant when num_shots > 0 and ensemble_size > 1.",
)

args = argparser.parse_args()

VAL_FILE = (
    "/home/animesh/storage/COLIEE-2023-Task-4/COLIEE2023statute_data-English/val.tsv"
)
TRAIN_FILE = (
    "/home/animesh/storage/COLIEE-2023-Task-4/COLIEE2023statute_data-English/train.tsv"
)
TEST_FILE = (
    "/home/animesh/storage/COLIEE-2023-Task-4/COLIEE2023statute_data-English/test.tsv"
)
TEXT_FILE = "/home/animesh/storage/COLIEE-2023-Task-4/COLIEE2023statute_data-English/text/civil_code_en-1to724-2.txt"
EXPLANATION_QUESTIONS = "/home/animesh/storage/COLIEE-2023-Task-4/COLIEE2023statute_data-English/train_withExplanationsAndQuestions.tsv"
model_results = args.model_name.replace("/", "-")
match args.balanced, args.similar_shots:
    case True, "yes":
        RESULTS_DIR = f"/home/animesh/storage/COLIEE-2023-Task-4/results/{model_results}_balanced_similar_shots"
    case True, "no":
        RESULTS_DIR = f"/home/animesh/storage/COLIEE-2023-Task-4/results/{model_results}_balanced_dissimilar_shots"
    case True, "mixed":
        RESULTS_DIR = f"/home/animesh/storage/COLIEE-2023-Task-4/results/{model_results}_balanced_mixed_shots"
    case False, "yes":
        RESULTS_DIR = f"/home/animesh/storage/COLIEE-2023-Task-4/results/{model_results}_similar_shots"
    case False, "no":
        RESULTS_DIR = f"/home/animesh/storage/COLIEE-2023-Task-4/results/{model_results}_dissimilar_shots"
    case False, "mixed":
        RESULTS_DIR = f"/home/animesh/storage/COLIEE-2023-Task-4/results/{model_results}_mixed_shots"

val = pandas.read_csv(VAL_FILE, sep="\t", index_col=0)
train = pandas.read_csv(TRAIN_FILE, sep="\t", index_col=0)
test = pandas.read_csv(TEST_FILE, sep="\t", index_col=0)
# set all cells under column "filename" to train
train["filename"] = "train"
train.set_index("pair_id", inplace=True)

articles = {}
with open(TEXT_FILE, "r") as f:
    for line in f.readlines()[4:]:  # skip first 4 lines
        line = line.strip()  # remove trailing whitespace
        if line.startswith("Article"):  # new article
            article_id = line.strip().split()[1]  # get article id
            articles[f"Article {article_id}"] = line + "\n"  # add article to dict
        else:
            articles[f"Article {article_id}"] += line + "\n"

# make RESULTS_DIR if it does not exist
if not os.path.exists(RESULTS_DIR):
    os.mkdir(RESULTS_DIR)

# check that model name is valid
match args.model_name:
    case "gpt-3.5-turbo" | "gpt-4":
        # if you want to use OpenAI, you MUST create an apikeys.py and set your open_ai API key variable
        import openai
        from apikeys import open_ai, org_id

        openai.api_key, openai.organization = open_ai, org_id
    case _:
        from transformers import AutoTokenizer, pipeline

        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        text_generation_pipeline = pipeline(model=args.model_name, device_map="auto")
if args.temperature_sensitive_ensemble:
    assert (
        args.ensemble_size > 1
    ), "Ensemble size must be > 1 when temperature-sensitive ensemble is used."
# check that splits are valid
for split in args.splits.split(","):
    assert split in ["H30", "R01", "R02", "R03", "Test", "train"], "Invalid split."
# check that prompting style is valid
assert args.prompting_style in [
    "vanilla",
    "select-most-relevant",
    "consider-both",
    "self-ask",
    "information-explanation",
], "Invalid prompting style."
# check that number of shots is valid
assert (
    0 <= args.num_shots <= len(train)
), f"Number of shots must be between 0 and the number of training examples ({len(train)})."
# check that temperature is valid
assert 0 <= args.temperature <= 1, "Temperature must be between 0 and 1."
# check that similarity metric is valid
assert args.similarity_metric in [
    None,
    "TFIDF",
    "sbert",
    "bleurt",
    "bert-score",
], "Invalid similarity metric."
# check that handleU is valid
if args.similarity_metric == "all":
    assert (
        args.similar_shots != "mixed"
    ), "Cannot use mixed similar shots when similarity metric is all."
assert args.handleU in [
    "random",
    "legal_terminology",
    "similar_articles",
    "similar_explanation",
    "similar_questions",
    "similar_articles_explanation",
    "similar_articles_questions",
    "similar_explanation_questions",
    "similar_articles_explanation_questions",
], "Invalid handleU."
match args.prompting_style:
    case "information-explanation":
        explanations_questions = pandas.read_csv(
            EXPLANATION_QUESTIONS, sep="\t", index_col=0
        )
        explanations_questions.set_index("pair_id", inplace=True)
        # replace articles, queries, and labels in explanations_questions from train based on pair_id
        for pair_id in explanations_questions.index:
            explanations_questions.loc[pair_id, "articles"] = train.loc[
                pair_id, "articles"
            ]
            explanations_questions.loc[pair_id, "query"] = train.loc[pair_id, "query"]
            explanations_questions.loc[pair_id, "label"] = train.loc[pair_id, "label"]
    case _:
        args.handleU = None
if args.temperature_sensitive_ensemble:
    args.ensemble_size = 5
print(args)

top_p = 1
lemmatizer = nltk.stem.WordNetLemmatizer()


# Define a function to preprocess text
def preprocess(text):
    return " ".join(
        [
            lemmatizer.lemmatize(word)  # lemmatize
            for word in nltk.word_tokenize(  # tokenize
                text.lower().translate(
                    str.maketrans("", "", string.punctuation)
                )  # remove punctuation, make lowercase
            )
            if word not in nltk.corpus.stopwords.words("english")  # remove stop words
        ]
    )


vectorizer, tfidf_matrix, train_list_for_similarity = None, None, None


def initialize_similarity():
    global vectorizer, tfidf_matrix, train_list_for_similarity
    # Preprocess the input strings and data strings
    match args.handleU:
        case "similar_explanation":
            if args.balanced:
                train_list_for_similarity = (
                    [
                        r["explanation"]
                        for _, r in explanations_questions.iterrows()
                        if r["label"] == "Y"
                    ],
                    [
                        r["explanation"]
                        for _, r in explanations_questions.iterrows()
                        if r["label"] == "N"
                    ],
                )
                preprocessed_train_list = (
                    [
                        preprocess(r["explanation"])
                        for _, r in explanations_questions.iterrows()
                        if r["label"] == "Y"
                    ],
                    [
                        preprocess(r["explanation"])
                        for _, r in explanations_questions.iterrows()
                        if r["label"] == "N"
                    ],
                )
            else:
                train_list_for_similarity = [
                    r["explanation"] for _, r in explanations_questions.iterrows()
                ]
                preprocessed_train_list = [
                    preprocess(r["explanation"])
                    for _, r in explanations_questions.iterrows()
                ]
        case "similar_questions":
            if args.balanced:
                train_list_for_similarity = (
                    [
                        r["questions"]
                        for _, r in explanations_questions.iterrows()
                        if r["label"] == "Y"
                    ],
                    [
                        r["questions"]
                        for _, r in explanations_questions.iterrows()
                        if r["label"] == "N"
                    ],
                )
                preprocessed_train_list = (
                    [
                        preprocess(r["questions"])
                        for _, r in explanations_questions.iterrows()
                        if r["label"] == "Y"
                    ],
                    [
                        preprocess(r["questions"])
                        for _, r in explanations_questions.iterrows()
                        if r["label"] == "N"
                    ],
                )
            else:
                train_list_for_similarity = [
                    r["questions"] for _, r in explanations_questions.iterrows()
                ]
                preprocessed_train_list = [
                    preprocess(r["questions"])
                    for _, r in explanations_questions.iterrows()
                ]
        case "similar_articles_explanation":
            if args.balanced:
                train_list_for_similarity = (
                    [
                        r["articles"] + "\n\n" + r["explanation"]
                        for _, r in explanations_questions.iterrows()
                        if r["label"] == "Y"
                    ],
                    [
                        r["articles"] + "\n\n" + r["explanation"]
                        for _, r in explanations_questions.iterrows()
                        if r["label"] == "N"
                    ],
                )
                preprocessed_train_list = (
                    [
                        preprocess(r["articles"] + "\n\n" + r["explanation"])
                        for _, r in explanations_questions.iterrows()
                        if r["label"] == "Y"
                    ],
                    [
                        preprocess(r["articles"] + "\n\n" + r["explanation"])
                        for _, r in explanations_questions.iterrows()
                        if r["label"] == "N"
                    ],
                )
            else:
                train_list_for_similarity = [
                    r["articles"] + "\n\n" + r["explanation"]
                    for _, r in explanations_questions.iterrows()
                ]
                preprocessed_train_list = [
                    preprocess(
                        r["articles"] + "\n\n" + r["query"] + "\n\n" + r["explanation"]
                    )
                    for _, r in explanations_questions.iterrows()
                ]
        case "similar_articles_questions":
            if args.balanced:
                train_list_for_similarity = (
                    [
                        r["articles"] + "\n\n" + r["questions"]
                        for _, r in explanations_questions.iterrows()
                        if r["label"] == "Y"
                    ],
                    [
                        r["articles"] + "\n\n" + r["questions"]
                        for _, r in explanations_questions.iterrows()
                        if r["label"] == "N"
                    ],
                )
                preprocessed_train_list = (
                    [
                        preprocess(r["articles"] + "\n\n" + r["questions"])
                        for _, r in explanations_questions.iterrows()
                        if r["label"] == "Y"
                    ],
                    [
                        preprocess(r["articles"] + "\n\n" + r["questions"])
                        for _, r in explanations_questions.iterrows()
                        if r["label"] == "N"
                    ],
                )
            else:
                train_list_for_similarity = [
                    r["articles"] + "\n\n" + r["questions"]
                    for _, r in explanations_questions.iterrows()
                ]
                preprocessed_train_list = [
                    preprocess(
                        r["articles"] + "\n\n" + r["query"] + "\n\n" + r["questions"]
                    )
                    for _, r in explanations_questions.iterrows()
                ]
        case "similar_explanation_questions":
            if args.balanced:
                train_list_for_similarity = (
                    [
                        r["explanation"] + "\n\n" + r["questions"]
                        for _, r in explanations_questions.iterrows()
                        if r["label"] == "Y"
                    ],
                    [
                        r["explanation"] + "\n\n" + r["questions"]
                        for _, r in explanations_questions.iterrows()
                        if r["label"] == "N"
                    ],
                )
                preprocessed_train_list = (
                    [
                        preprocess(r["explanation"] + "\n\n" + r["questions"])
                        for _, r in explanations_questions.iterrows()
                        if r["label"] == "Y"
                    ],
                    [
                        preprocess(r["explanation"] + "\n\n" + r["questions"])
                        for _, r in explanations_questions.iterrows()
                        if r["label"] == "N"
                    ],
                )
            else:
                train_list_for_similarity = [
                    r["explanation"] + "\n\n" + r["questions"]
                    for _, r in explanations_questions.iterrows()
                ]
                preprocessed_train_list = [
                    preprocess(r["explanation"] + "\n\n" + r["questions"])
                    for _, r in explanations_questions.iterrows()
                ]
        case "similar_articles_explanation_questions":
            if args.balanced:
                train_list_for_similarity = (
                    [
                        r["articles"]
                        + "\n\n"
                        + r["explanation"]
                        + "\n\n"
                        + r["questions"]
                        for _, r in explanations_questions.iterrows()
                        if r["label"] == "Y"
                    ],
                    [
                        r["articles"]
                        + "\n\n"
                        + r["explanation"]
                        + "\n\n"
                        + r["questions"]
                        for _, r in explanations_questions.iterrows()
                        if r["label"] == "N"
                    ],
                )
                preprocessed_train_list = (
                    [
                        preprocess(
                            r["articles"]
                            + "\n\n"
                            + r["explanation"]
                            + "\n\n"
                            + r["questions"]
                        )
                        for _, r in explanations_questions.iterrows()
                        if r["label"] == "Y"
                    ],
                    [
                        preprocess(
                            r["articles"]
                            + "\n\n"
                            + r["explanation"]
                            + "\n\n"
                            + r["questions"]
                        )
                        for _, r in explanations_questions.iterrows()
                        if r["label"] == "N"
                    ],
                )
            else:
                train_list_for_similarity = [
                    r["articles"] + "\n\n" + r["explanation"] + "\n\n" + r["questions"]
                    for _, r in explanations_questions.iterrows()
                ]
                preprocessed_train_list = [
                    preprocess(
                        r["articles"]
                        + "\n\n"
                        + r["query"]
                        + "\n\n"
                        + r["explanation"]
                        + "\n\n"
                        + r["questions"]
                    )
                    for _, r in explanations_questions.iterrows()
                ]
        case _:
            if args.balanced:
                train_list_for_similarity = (
                    [
                        r["articles"] + "\n\n" + r["query"]
                        for _, r in train.iterrows()
                        if r["label"] == "Y"
                    ],
                    [
                        r["articles"] + "\n\n" + r["query"]
                        for _, r in train.iterrows()
                        if r["label"] == "N"
                    ],
                )
                preprocessed_train_list = (
                    [
                        preprocess(r["articles"] + "\n\n" + r["query"])
                        for _, r in train.iterrows()
                        if r["label"] == "Y"
                    ],
                    [
                        preprocess(r["articles"] + "\n\n" + r["query"])
                        for _, r in train.iterrows()
                        if r["label"] == "N"
                    ],
                )
            else:
                train_list_for_similarity = [
                    r["articles"] + "\n\n" + r["query"] for _, r in train.iterrows()
                ]
                preprocessed_train_list = [
                    preprocess(r["articles"] + "\n\n" + r["query"])
                    for _, r in train.iterrows()
                ]
    # Create a TF-IDF vectorizer and fit it to the preprocessed input and data strings
    if args.balanced:
        vectorizer = (TfidfVectorizer(), TfidfVectorizer())
        tfidf_matrix = (
            vectorizer[0].fit_transform(preprocessed_train_list[0]),
            vectorizer[1].fit_transform(preprocessed_train_list[1]),
        )
    else:
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(preprocessed_train_list)


# Define a function to get the most similar strings
def get_most_similar(input_string, count=5, dissimilar=False):
    if vectorizer is None or tfidf_matrix is None or train_list_for_similarity is None:
        initialize_similarity()
    if args.balanced:
        train_vectors_N, train_vectors_Y = (
            tfidf_matrix[0],
            tfidf_matrix[1],
        )  # Transform the preprocessed input strings and data strings into TF-IDF vectors
        val_vector_N, val_vector_Y = vectorizer[0].transform(
            [preprocess(input_string)]
        ), vectorizer[1].transform([preprocess(input_string)])
        cosine_similarities_N, cosine_similarities_Y = cosine_similarity(
            val_vector_N, train_vectors_N
        ), cosine_similarity(
            val_vector_Y, train_vectors_Y
        )  # Calculate the cosine similarities between the input vectors and the data vectors
        if dissimilar:
            most_similar_indices_N, most_similar_indices_Y = (
                cosine_similarities_N.argsort()[:, : count // 2],
                cosine_similarities_Y.argsort()[:, : count // 2],
            )  # Get the indices of the most similar strings for each input vector
        else:
            most_similar_indices_N, most_similar_indices_Y = (
                cosine_similarities_N.argsort()[:, ::-1][:, : count // 2],
                cosine_similarities_Y.argsort()[:, ::-1][:, : count // 2],
            )  # Get the indices of the most similar strings for each input vector
        most_similar_indices = numpy.concatenate(
            (most_similar_indices_N[0], most_similar_indices_Y[0])
        )
    else:
        train_vectors = tfidf_matrix
        val_vector = vectorizer.transform([preprocess(input_string)])
        cosine_similarities = cosine_similarity(
            val_vector, train_vectors
        )  # Calculate the cosine similarities between the input vectors and the data vectors
        if dissimilar:
            most_similar_indices = cosine_similarities.argsort()[
                :, :count
            ]  # Get the indices of the most similar strings for each input vector
        else:
            most_similar_indices = cosine_similarities.argsort()[:, ::-1][
                :, :count
            ]  # Get the indices of the most similar strings for each input vector
        most_similar_indices = most_similar_indices[0]
    # Get the most similar strings for each input vector
    match args.prompting_style:
        case "information-explanation":
            return [
                (
                    f"\nArticles:\n{explanations_questions['articles'].iloc[i]}\n\nQuery:\n{explanations_questions['query'].iloc[i]}",
                    explanations_questions["label"].iloc[i],
                )
                for i in most_similar_indices
            ]
        case _:
            return [
                (
                    f"\nArticles:\n{train['articles'].iloc[i]}\n\nQuery:\n{train['query'].iloc[i]}",
                    train["label"].iloc[i],
                )
                for i in most_similar_indices
            ]


def replace_article_ids(msg_articles):
    for article_name in re.findall(r"Article \d+", msg_articles):  # for each article
        try:  # try to get article text
            msg_articles = (
                articles[article_name] + "\n" + msg_articles
            )  # append article to prompt
        except KeyError:  # if article text not found
            msg_articles = msg_articles.replace(
                article_name, ""
            )  # remove article name from prompt
    return msg_articles


def add_num_shots_to_prompt(row, msg, num_shots):
    if vectorizer is None or tfidf_matrix is None or train_list_for_similarity is None:
        initialize_similarity()
    match args.similarity_metric:
        case "TFIDF":  # if using TFIDF
            match args.similar_shots:
                case "yes":
                    if "train" in args.splits.split(","):  # if training split is used
                        similar_examples = get_most_similar(
                            row["articles"] + "\n\n" + row["query"], args.num_shots
                        )[
                            1:
                        ]  # get most similar examples
                    else:  # if training split is not used
                        similar_examples = get_most_similar(
                            row["articles"] + "\n\n" + row["query"], args.num_shots
                        )  # get most similar examples
                case "no":
                    if "train" in args.splits.split(","):  # if training split is used
                        similar_examples = get_most_similar(
                            row["articles"] + "\n\n" + row["query"],
                            args.num_shots,
                            dissimilar=True,
                        )[
                            1:
                        ]  # get most dissimilar examples
                    else:  # if training split is not used
                        similar_examples = get_most_similar(
                            row["articles"] + "\n\n" + row["query"],
                            args.num_shots,
                            dissimilar=True,
                        )  # get most dissimilar examples
                case "mixed":
                    if "train" in args.splits.split(","):  # if training split is used
                        similar_examples = (
                            get_most_similar(
                                row["articles"] + "\n\n" + row["query"],
                                args.num_shots // 2,
                            )[1:]
                            + get_most_similar(
                                row["articles"] + "\n\n" + row["query"],
                                args.num_shots // 2,
                                dissimilar=True,
                            )[1:]
                        )  # get most similar and most dissimilar examples
                    else:  # if training split is not used
                        similar_examples = get_most_similar(
                            row["articles"] + "\n\n" + row["query"], args.num_shots // 2
                        ) + get_most_similar(
                            row["articles"] + "\n\n" + row["query"],
                            args.num_shots // 2,
                            dissimilar=True,
                        )  # get most similar and most dissimilar examples
            for example in similar_examples:  # for each example
                msg.append(
                    {"role": "user", "content": example[0]}
                )  # append example to prompt
                msg.append(
                    {"role": "assistant", "content": example[1]}
                )  # append example label to prompt
        case "sbert" | "bleurt" | "bert-score":
            if args.balanced:
                match args.similar_shots:
                    case "yes":
                        if "train" in args.splits.split(
                            ","
                        ):  # if training split is used
                            similar_indices = (
                                similarity.get_similarity(
                                    model_name=args.similarity_metric,
                                    sentences1=train_list_for_similarity[0],
                                    sentences2=[
                                        row["articles"] + "\n\n" + row["query"]
                                    ],
                                    return_most_similar=args.num_shots // 2,
                                )[1:]
                                + similarity.get_similarity(
                                    model_name=args.similarity_metric,
                                    sentences1=train_list_for_similarity[1],
                                    sentences2=[
                                        row["articles"] + "\n\n" + row["query"]
                                    ],
                                    return_most_similar=args.num_shots // 2,
                                )[1:]
                            )
                        else:  # if training split is not used
                            similar_indices = similarity.get_similarity(
                                model_name=args.similarity_metric,
                                sentences1=train_list_for_similarity[0],
                                sentences2=[row["articles"] + "\n\n" + row["query"]],
                                return_most_similar=args.num_shots // 2,
                            ) + similarity.get_similarity(
                                model_name=args.similarity_metric,
                                sentences1=train_list_for_similarity[1],
                                sentences2=[row["articles"] + "\n\n" + row["query"]],
                                return_most_similar=args.num_shots // 2,
                            )
                    case "no":
                        if "train" in args.splits.split(
                            ","
                        ):  # if training split is used
                            similar_indices = (
                                similarity.get_similarity(
                                    model_name=args.similarity_metric,
                                    sentences1=train_list_for_similarity[0],
                                    sentences2=[
                                        row["articles"] + "\n\n" + row["query"]
                                    ],
                                    return_most_dissimilar=args.num_shots // 2,
                                )[1:]
                                + similarity.get_similarity(
                                    model_name=args.similarity_metric,
                                    sentences1=train_list_for_similarity[1],
                                    sentences2=[
                                        row["articles"] + "\n\n" + row["query"]
                                    ],
                                    return_most_dissimilar=args.num_shots // 2,
                                )[1:]
                            )
                        else:  # if training split is not used
                            similar_indices = similarity.get_similarity(
                                model_name=args.similarity_metric,
                                sentences1=train_list_for_similarity[0],
                                sentences2=[row["articles"] + "\n\n" + row["query"]],
                                return_most_dissimilar=args.num_shots // 2,
                            ) + similarity.get_similarity(
                                model_name=args.similarity_metric,
                                sentences1=train_list_for_similarity[1],
                                sentences2=[row["articles"] + "\n\n" + row["query"]],
                                return_most_dissimilar=args.num_shots // 2,
                            )
                    case "mixed":
                        if "train" in args.splits.split(
                            ","
                        ):  # if training split is used
                            similar_indices = (
                                similarity.get_similarity(
                                    model_name=args.similarity_metric,
                                    sentences1=train_list_for_similarity[0],
                                    sentences2=[
                                        row["articles"] + "\n\n" + row["query"]
                                    ],
                                    return_most_similar=args.num_shots // 4,
                                )[1:]
                                + similarity.get_similarity(
                                    model_name=args.similarity_metric,
                                    sentences1=train_list_for_similarity[1],
                                    sentences2=[
                                        row["articles"] + "\n\n" + row["query"]
                                    ],
                                    return_most_similar=args.num_shots // 4,
                                )[1:]
                                + similarity.get_similarity(
                                    model_name=args.similarity_metric,
                                    sentences1=train_list_for_similarity[0],
                                    sentences2=[
                                        row["articles"] + "\n\n" + row["query"]
                                    ],
                                    return_most_dissimilar=args.num_shots // 4,
                                )[1:]
                                + similarity.get_similarity(
                                    model_name=args.similarity_metric,
                                    sentences1=train_list_for_similarity[1],
                                    sentences2=[
                                        row["articles"] + "\n\n" + row["query"]
                                    ],
                                    return_most_dissimilar=args.num_shots // 4,
                                )[1:]
                            )
                        else:  # if training split is not used
                            similar_indices = (
                                similarity.get_similarity(
                                    model_name=args.similarity_metric,
                                    sentences1=train_list_for_similarity[0],
                                    sentences2=[
                                        row["articles"] + "\n\n" + row["query"]
                                    ],
                                    return_most_similar=args.num_shots // 4,
                                )
                                + similarity.get_similarity(
                                    model_name=args.similarity_metric,
                                    sentences1=train_list_for_similarity[1],
                                    sentences2=[
                                        row["articles"] + "\n\n" + row["query"]
                                    ],
                                    return_most_similar=args.num_shots // 4,
                                )
                                + similarity.get_similarity(
                                    model_name=args.similarity_metric,
                                    sentences1=train_list_for_similarity[0],
                                    sentences2=[
                                        row["articles"] + "\n\n" + row["query"]
                                    ],
                                    return_most_dissimilar=args.num_shots // 4,
                                )
                                + similarity.get_similarity(
                                    model_name=args.similarity_metric,
                                    sentences1=train_list_for_similarity[1],
                                    sentences2=[
                                        row["articles"] + "\n\n" + row["query"]
                                    ],
                                    return_most_dissimilar=args.num_shots // 4,
                                )
                            )
            else:
                match args.similar_shots:
                    case "yes":
                        if "train" in args.splits.split(
                            ","
                        ):  # if training split is used
                            similar_indices = similarity.get_similarity(
                                model_name=args.similarity_metric,
                                sentences1=train_list_for_similarity,
                                sentences2=[row["articles"] + "\n\n" + row["query"]],
                                return_most_similar=args.num_shots,
                            )[1:]
                        else:  # if training split is not used
                            similar_indices = similarity.get_similarity(
                                model_name=args.similarity_metric,
                                sentences1=train_list_for_similarity,
                                sentences2=[row["articles"] + "\n\n" + row["query"]],
                                return_most_similar=args.num_shots,
                            )
                    case "no":
                        if "train" in args.splits.split(
                            ","
                        ):  # if training split is used
                            similar_indices = similarity.get_similarity(
                                model_name=args.similarity_metric,
                                sentences1=train_list_for_similarity,
                                sentences2=[row["articles"] + "\n\n" + row["query"]],
                                return_most_dissimilar=args.num_shots,
                            )[1:]
                        else:  # if training split is not used
                            similar_indices = similarity.get_similarity(
                                model_name=args.similarity_metric,
                                sentences1=train_list_for_similarity,
                                sentences2=[row["articles"] + "\n\n" + row["query"]],
                                return_most_dissimilar=args.num_shots,
                            )
                    case "mixed":
                        if "train" in args.splits.split(
                            ","
                        ):  # if training split is used
                            similar_indices = (
                                similarity.get_similarity(
                                    model_name=args.similarity_metric,
                                    sentences1=train_list_for_similarity,
                                    sentences2=[
                                        row["articles"] + "\n\n" + row["query"]
                                    ],
                                    return_most_similar=args.num_shots // 2,
                                )[1:]
                                + similarity.get_similarity(
                                    model_name=args.similarity_metric,
                                    sentences1=train_list_for_similarity,
                                    sentences2=[
                                        row["articles"] + "\n\n" + row["query"]
                                    ],
                                    return_most_dissimilar=args.num_shots // 2,
                                )[1:]
                            )
                        else:  # if training split is not used
                            similar_indices = similarity.get_similarity(
                                model_name=args.similarity_metric,
                                sentences1=train_list_for_similarity,
                                sentences2=[row["articles"] + "\n\n" + row["query"]],
                                return_most_similar=args.num_shots // 2,
                            ) + similarity.get_similarity(
                                model_name=args.similarity_metric,
                                sentences1=train_list_for_similarity,
                                sentences2=[row["articles"] + "\n\n" + row["query"]],
                                return_most_dissimilar=args.num_shots // 2,
                            )
            similar_examples = [
                (
                    f"\nArticles:\n{train['articles'].iloc[i]}\n\nQuery:\n{train['query'].iloc[i]}",
                    train["label"].iloc[i],
                )
                for i in similar_indices
            ]
            for example in similar_examples:  # for each example
                msg.append(
                    {"role": "user", "content": example[0]}
                )  # append example to prompt
                msg.append(
                    {"role": "assistant", "content": example[1]}
                )  # append example label to prompt
        case "all":
            similar_indices = []
            for similarity_metric in ["TFIDF", "sbert", "bleurt", "bert-score"]:
                if similarity_metric == "TFIDF":
                    match args.similar_shots:
                        case "yes":
                            if "train" in args.splits.split(
                                ","
                            ):  # if training split is used
                                similar_indices += get_most_similar(
                                    row["articles"] + "\n\n" + row["query"],
                                    args.num_shots // 4,
                                )[1:]
                            else:  # if training split is not used
                                similar_indices += get_most_similar(
                                    row["articles"] + "\n\n" + row["query"],
                                    args.num_shots // 4,
                                )
                        case "no":
                            if "train" in args.splits.split(
                                ","
                            ):  # if training split is used
                                similar_indices += get_most_similar(
                                    row["articles"] + "\n\n" + row["query"],
                                    args.num_shots // 4,
                                    dissimilar=True,
                                )[1:]
                            else:  # if training split is not used
                                similar_indices += get_most_similar(
                                    row["articles"] + "\n\n" + row["query"],
                                    args.num_shots // 4,
                                    dissimilar=True,
                                )
                else:
                    match args.similar_shots:
                        case "yes":
                            if "train" in args.splits.split(
                                ","
                            ):  # if training split is used
                                similar_indices += similarity.get_similarity(
                                    model_name=similarity_metric,
                                    sentences1=train_list_for_similarity,
                                    sentences2=[
                                        row["articles"] + "\n\n" + row["query"]
                                    ],
                                    return_most_similar=args.num_shots // 4,
                                )[1:]
                            else:  # if training split is not used
                                similar_indices += similarity.get_similarity(
                                    model_name=similarity_metric,
                                    sentences1=train_list_for_similarity,
                                    sentences2=[
                                        row["articles"] + "\n\n" + row["query"]
                                    ],
                                    return_most_similar=args.num_shots // 4,
                                )
                        case "no":
                            if "train" in args.splits.split(
                                ","
                            ):  # if training split is used
                                similar_indices += similarity.get_similarity(
                                    model_name=similarity_metric,
                                    sentences1=train_list_for_similarity,
                                    sentences2=[
                                        row["articles"] + "\n\n" + row["query"]
                                    ],
                                    return_most_dissimilar=args.num_shots // 4,
                                )[1:]
                            else:  # if training split is not used
                                similar_indices += similarity.get_similarity(
                                    model_name=similarity_metric,
                                    sentences1=train_list_for_similarity,
                                    sentences2=[
                                        row["articles"] + "\n\n" + row["query"]
                                    ],
                                    return_most_dissimilar=args.num_shots // 4,
                                )
            for i in similar_indices:
                example = (
                    f"\nArticles:\n{train['articles'].iloc[i]}\n\nQuery:\n{train['query'].iloc[i]}",
                    train["label"].iloc[i],
                )
                msg.append(
                    {"role": "user", "content": example[0]}
                )  # append example to prompt
                msg.append(
                    {"role": "assistant", "content": example[1]}
                )  # append example label to prompt
        case _:
            for i in random.sample(range(len(train)), num_shots):  # for each example
                msg.append(
                    {
                        "role": "user",
                        "content": f"\nArticles:\n{replace_article_ids(train.iloc[i]['articles'])}\n\nQuery:\n{train.iloc[i]['query']}",
                    }
                )  # append example to prompt
                msg.append(
                    {"role": "assistant", "content": train.iloc[i]["label"]}
                )  # append example label to prompt
    return msg


def handleUncertain(msg, row, legal_terms, questions, explanation):
    match args.handleU:
        case "legal_terminology":
            similar_examples = get_most_similar(legal_terms, 10)
        case "similar_explanation":
            similar_examples = get_most_similar(explanation, 10)
        case "similar_questions":
            similar_examples = get_most_similar(questions, 10)
        case "similar_articles_explanation":
            similar_examples = get_most_similar(
                row["articles"] + "\n\n" + row["query"] + "\n\n" + explanation, 10
            )
        case "similar_articles_questions":
            similar_examples = get_most_similar(
                row["articles"] + "\n\n" + row["query"] + "\n\n" + questions, 10
            )
        case "similar_explanation_questions":
            similar_examples = get_most_similar(explanation + "\n\n" + questions, 10)
        case "similar_articles_explanation_questions":
            similar_examples = get_most_similar(
                row["articles"]
                + "\n\n"
                + row["query"]
                + "\n\n"
                + explanation
                + "\n\n"
                + questions,
                10,
            )
        case _:
            similar_examples = get_most_similar(
                row["articles"] + "\n\n" + row["query"], 10
            )  # get most similar examples
    for example in [
        similar_examples[i] for i in random.sample(range(len(similar_examples)), 3)
    ]:  # for each example
        msg.append({"role": "user", "content": example[0]})  # append example to prompt
        msg.append(
            {"role": "assistant", "content": example[1]}
        )  # append example label to prompt
    return msg


def get_output(msg, temperature, max_tokens):
    match args.model_name:
        case "gpt-3.5-turbo" | "gpt-4":
            while True:
                try:
                    output = openai.ChatCompletion.create(  # get model output
                        model=args.model_name,  # use turbo model
                        messages=msg,  # use prompt
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                    )["choices"][0]["message"]["content"]
                    return output
                except openai.error.InvalidRequestError:
                    if max_tokens > 32:
                        max_tokens //= 2
                        continue
                    else:
                        msg = msg[:2] + msg[3:]  # remove third message
                        if len(msg) < 4:
                            raise
                    continue
        case _:
            while True:
                text = ""
                for m in msg:
                    if m["role"] == "assistant":
                        text += "Answer: " + m["content"] + "\n#####\n"
                    else:
                        text += m["content"] + "\n#####\n"
                text += "Answer:"
                if len(tokenizer(text)) > tokenizer.model_max_length:
                    msg = msg[:2] + msg[3:]  # remove third message
                    if len(msg) < 4:
                        raise ValueError("Prompt too long.")
                    continue
                return text_generation_pipeline(
                    text,
                    max_length=2048,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=args.do_sample,
                    num_return_sequences=1,
                )[0]["generated_text"]


color = random.choice(
    ["red", "green", "yellow", "blue", "magenta", "cyan"]
)  # get random color for printing
df = val
if "Test" in args.splits.split(","):  # if test split
    df = pandas.concat([df, test])  # combine val and test
if "train" in args.splits.split(","):  # if training split
    df = pandas.concat([df, train])  # combine val and test
for split in args.splits.split(","):  # for each split
    print(colored(split, color))
    match args.prompting_style:
        case "information-explanation":
            results_file = f"{RESULTS_DIR}/information-explanation/{split}_{args.handleU}_{args.num_shots}_2_{args.ensemble_size}.tsv"
            raw_results_file = f"{RESULTS_DIR}/information-explanation/{split}_{args.handleU}_{args.num_shots}_2_{args.ensemble_size}_raw.tsv"
        case _:
            match args.similarity_metric, args.temperature_sensitive_ensemble:
                case None, True:
                    results_file = f"{RESULTS_DIR}/{split}_{args.prompting_style}_{args.num_shots}_2_{args.ensemble_size}.tsv"
                    raw_results_file = f"{RESULTS_DIR}/{split}_{args.prompting_style}_{args.num_shots}_2_{args.ensemble_size}_raw.tsv"
                case None, False:
                    results_file = f"{RESULTS_DIR}/{split}_{args.prompting_style}_{args.num_shots}_{args.temperature}_{args.ensemble_size}.tsv"
                    raw_results_file = f"{RESULTS_DIR}/{split}_{args.prompting_style}_{args.num_shots}_{args.temperature}_{args.ensemble_size}_raw.tsv"
                case _, True:
                    results_file = f"{RESULTS_DIR}/{split}_{args.prompting_style}_{args.num_shots}_2_{args.ensemble_size}_{args.similarity_metric}.tsv"
                    raw_results_file = f"{RESULTS_DIR}/{split}_{args.prompting_style}_{args.num_shots}_2_{args.ensemble_size}_{args.similarity_metric}_raw.tsv"
                case _, False:
                    results_file = f"{RESULTS_DIR}/{split}_{args.prompting_style}_{args.num_shots}_{args.temperature}_{args.ensemble_size}_{args.similarity_metric}.tsv"
                    raw_results_file = f"{RESULTS_DIR}/{split}_{args.prompting_style}_{args.num_shots}_{args.temperature}_{args.ensemble_size}_{args.similarity_metric}_raw.tsv"
    if args.overwrite:
        results = pandas.DataFrame(
            columns=["pair_id", "label", "prediction", "votes"]
        )  # create empty dataframe to store results
        raw_results = pandas.DataFrame(
            columns=["pair_id", "label", "prediction", "votes", "msg", "final_output"]
        )  # create empty dataframe to store results
    else:
        if os.path.exists(raw_results_file) and os.path.exists(
            results_file
        ):  # if results file exists
            results = pandas.read_csv(results_file, sep="\t")  # load results
            raw_results = pandas.read_csv(raw_results_file, sep="\t")  # load results
            if len(df[df["filename"].str.contains(split)]) == len(
                results
            ):  # if results are complete
                print(
                    colored(
                        f"Skipping split. Got {sum(results['label'] == results['prediction'])}/{len(results)} items right. "
                        f"Accuracy is {sum(results['label'] == results['prediction']) / len(results)}",
                        color,
                    )
                )
                continue  # skip split
        else:
            results = pandas.DataFrame(
                columns=["pair_id", "label", "prediction", "votes"]
            )  # create empty dataframe to store results
            raw_results = pandas.DataFrame(
                columns=[
                    "pair_id",
                    "label",
                    "prediction",
                    "votes",
                    "msg",
                    "final_output",
                ]
            )  # create empty dataframe to store results

    try:
        pbar = tqdm(
            total=len(df[df["filename"].str.contains(split)])
        )  # create progress bar
        for index, row in df[
            df["filename"].str.contains(split)
        ].iterrows():  # for each example
            if row["pair_id"] in results["pair_id"].values:  # if example already done
                pbar.update(1)  # update progress bar
                continue  # skip example
            match args.prompting_style:
                case "information-explanation":  # information-explanation prompting style
                    PATIENCE = True  # set patience to True
                    legal_terms, questions, explanation = (
                        None,
                        None,
                        None,
                    )  # initialize legal terms, questions, and explanation
                    while True:
                        votes = []  # initialize votes
                        for e in range(
                            args.ensemble_size
                        ):  # for each model in ensemble
                            msg = [
                                {
                                    "role": "system",
                                    "content": "You are a legal reasoning system.",
                                }
                            ]  # initialize message
                            if not PATIENCE:
                                if legal_terms and questions and explanation:
                                    msg = handleUncertain(
                                        msg, row, legal_terms, questions, explanation
                                    )
                                else:
                                    raise ValueError(
                                        "Legal terms, questions, and explanation must be defined."
                                    )
                            msg.append(
                                {
                                    "role": "user",
                                    "content": f"\n\nArticles:\n{replace_article_ids(row['articles'])}\n\nQuery:\n{row['query']}\n\nPlease write a comma-separated list of the legal terminology contained in the articles and query.",
                                }
                            )
                            legal_terms = (
                                get_output(msg, e / (args.ensemble_size - 1), 1024)
                                if args.ensemble_size > 1
                                else get_output(msg, args.temperature, 1024)
                            )
                            msg.append({"role": "assistant", "content": legal_terms})
                            msg.append(
                                {
                                    "role": "user",
                                    "content": "In order to determine whether the query follows logically from the articles, which questions should we answer first?",
                                }
                            )
                            questions = (
                                get_output(msg, e / (args.ensemble_size - 1), 1024)
                                if args.ensemble_size > 1
                                else get_output(msg, args.temperature, 1024)
                            )
                            msg.append({"role": "assistant", "content": questions})
                            msg.append(
                                {
                                    "role": "user",
                                    "content": "What are the answers to those questions?",
                                }
                            )
                            answers = (
                                get_output(msg, e / (args.ensemble_size - 1), 1024)
                                if args.ensemble_size > 1
                                else get_output(msg, args.temperature, 1024)
                            )
                            msg.append({"role": "assistant", "content": answers})
                            if PATIENCE:
                                msg.append(
                                    {
                                        "role": "user",
                                        "content": "Given the above, does the query follow from the articles? Please first explain your reasoning, then say 'Y' for yes, 'N' for no, or 'U' if there is not enough information in the articles to determine. Please enclose your final answer 'Y', 'N', or 'U' in single quotes.",
                                    }
                                )
                            else:
                                msg.append(
                                    {
                                        "role": "user",
                                        "content": "Given the above, does the query follow from the articles? Please first explain your reasoning, then say 'Y' for yes and 'N' for no. You have had enough time to ask for more information, now you have to pick 'Y' or 'N'. Please enclose your final answer 'Y' or 'N' in single quotes.",
                                    }
                                )
                            explanation = (
                                get_output(msg, e / (args.ensemble_size - 1), 1024)
                                if args.ensemble_size > 1
                                else get_output(msg, args.temperature, 1024)
                            )
                            if "'Y'" in explanation:
                                votes.append("Y")
                                raw_results = pandas.concat(
                                    [
                                        raw_results,
                                        pandas.DataFrame(
                                            {
                                                "pair_id": [row["pair_id"]],
                                                "label": [row["label"]],
                                                "votes": [votes],
                                                "msg": [msg],
                                                "final_output": [explanation],
                                            }
                                        ),
                                    ]
                                )
                            elif "'N'" in explanation:
                                votes.append("N")
                                raw_results = pandas.concat(
                                    [
                                        raw_results,
                                        pandas.DataFrame(
                                            {
                                                "pair_id": [row["pair_id"]],
                                                "label": [row["label"]],
                                                "votes": [votes],
                                                "msg": [msg],
                                                "final_output": [explanation],
                                            }
                                        ),
                                    ]
                                )
                            elif "'U'" in explanation and PATIENCE:
                                votes.append("U")
                                raw_results = pandas.concat(
                                    [
                                        raw_results,
                                        pandas.DataFrame(
                                            {
                                                "pair_id": [row["pair_id"]],
                                                "label": [row["label"]],
                                                "votes": [votes],
                                                "msg": [msg],
                                                "final_output": [explanation],
                                            }
                                        ),
                                    ]
                                )
                            else:
                                msg.append(
                                    {"role": "assistant", "content": explanation}
                                )  # append model output to prompt
                                if PATIENCE:
                                    msg.append(
                                        {
                                            "role": "user",
                                            "content": "What is the answer? Please just say 'Y', 'N', or 'U' enclosed in single quotes. Do not say ANYTHING else.",
                                        }
                                    )  # append final question to prompt
                                else:
                                    msg.append(
                                        {
                                            "role": "user",
                                            "content": "What is the answer? You have had enough time to ask for more information, now you have to pick 'Y' or 'N'. Please just say 'Y' or 'N' enclosed in single quotes. Do not say ANYTHING else.",
                                        }
                                    )
                                output = (
                                    get_output(msg, e / (args.ensemble_size - 1), 1024)
                                    if args.ensemble_size > 1
                                    else get_output(msg, args.temperature, 1024)
                                )
                                if "'Y'" in output:
                                    votes.append("Y")
                                elif "'N'" in output:
                                    votes.append("N")
                                else:
                                    if len(output.strip()) == 1:
                                        votes.append(output)
                                    else:
                                        votes.append("U")
                                raw_results = pandas.concat(
                                    [
                                        raw_results,
                                        pandas.DataFrame(
                                            {
                                                "pair_id": [row["pair_id"]],
                                                "label": [row["label"]],
                                                "votes": [votes],
                                                "msg": [msg],
                                                "final_output": [output],
                                            }
                                        ),
                                    ]
                                )
                        prediction = (
                            "U"
                            if votes.count("Y") == votes.count("N")
                            else Counter(votes).most_common(1)[0][0]
                        )
                        if prediction == "U":
                            if args.handleU == "random":
                                break
                            else:
                                PATIENCE = False
                                continue
                        else:
                            break
                    if prediction == "U":
                        prediction = random.choice(["Y", "N"])
                    results = pandas.concat(
                        [
                            results,
                            pandas.DataFrame(
                                {
                                    "pair_id": [row["pair_id"]],
                                    "label": [row["label"]],
                                    "votes": [votes],
                                    "prediction": [prediction],
                                }
                            ),
                        ]
                    )
                    pbar.update(1)  # update progress bar
                    results.to_csv(results_file, index=False, sep="\t")  # save results
                    raw_results.to_csv(
                        raw_results_file, index=False, sep="\t"
                    )  # save results
                case _:
                    votes = []  # create empty list to store votes
                    for e in range(args.ensemble_size):  # for each model in ensemble
                        num_shots = args.num_shots
                        match args.prompting_style:  # match prompting style
                            case "vanilla":  # vanilla
                                msg = [
                                    {
                                        "role": "system",
                                        "content": "You are a legal reasoning system. Given a set of relevant articles, you must answer whether a legal statement is true. Just answer as 'Y' for yes and 'N' for no enclosed in single quotes. Write nothing else.",
                                    }
                                ]
                                if args.num_shots > 1:  # if using shots
                                    msg = add_num_shots_to_prompt(
                                        row, msg, num_shots
                                    )  # add shots to prompt
                                msg.append(
                                    {
                                        "role": "user",
                                        "content": f"Articles:\n{replace_article_ids(row['articles'])}\n\nQuery:\n{row['query']}\n\nJust answer as 'Y' for yes and 'N' for no enclosed in single quotes. Write nothing else.",
                                    }
                                )  # append query to prompt
                                if (
                                    args.temperature_sensitive_ensemble
                                ):  # if using temperature-sensitive ensemble
                                    output = get_output(
                                        msg, e / (args.ensemble_size - 1), 1024
                                    )  # get model output
                                else:  # if not using temperature-sensitive ensemble
                                    output = get_output(
                                        msg, args.temperature, 1024
                                    )  # get model output
                                if "'Y'" in output:
                                    votes.append("Y")
                                elif "'N'" in output:
                                    votes.append("N")
                                else:
                                    if len(output.strip()) == 1:
                                        votes.append(output)
                                    else:
                                        votes.append("U")
                                raw_results = pandas.concat(
                                    [
                                        raw_results,
                                        pandas.DataFrame(
                                            {
                                                "pair_id": [row["pair_id"]],
                                                "label": [row["label"]],
                                                "votes": [votes],
                                                "msg": [msg],
                                                "final_output": [output],
                                            }
                                        ),
                                    ]
                                )

                            case "select-most-relevant":  # select most relevant
                                msg = [
                                    {
                                        "role": "system",
                                        "content": "You are a legal reasoning system. Given a set of relevant articles, you must answer whether a legal statement is true.",
                                    }
                                ]
                                if args.num_shots > 1:  # if using shots
                                    msg = add_num_shots_to_prompt(
                                        row, msg, num_shots
                                    )  # add shots to prompt
                                msg.append(
                                    {
                                        "role": "user",
                                        "content": f"\n\nArticles:\n{replace_article_ids(row['articles'])}\n\nQuery:\n{row['query']}\n\nWhich article(s) do you think are relevant for answering the query?",
                                    }
                                )  # append prompt to prompt
                                msg.append(
                                    {
                                        "role": "assistant",
                                        "content": get_output(msg, 1, 1024),
                                    }
                                )  # append model output to prompt
                                msg.append(
                                    {
                                        "role": "user",
                                        "content": "Answer the query based on these articles. Just answer as 'Y' for yes and 'N' for no enclosed in single quotes. Write nothing else.",
                                    }
                                )  # append final question to prompt
                                if (
                                    args.temperature_sensitive_ensemble
                                ):  # if using temperature-sensitive ensemble
                                    output = get_output(
                                        msg, e / (args.ensemble_size - 1), 1024
                                    )  # get model output
                                else:  # if not using temperature-sensitive ensemble
                                    output = get_output(
                                        msg, args.temperature, 1024
                                    )  # get model output
                                if "'Y'" in output:
                                    votes.append("Y")
                                elif "'N'" in output:
                                    votes.append("N")
                                else:
                                    if len(output.strip()) == 1:
                                        votes.append(output)
                                    else:
                                        votes.append("U")
                                raw_results = pandas.concat(
                                    [
                                        raw_results,
                                        pandas.DataFrame(
                                            {
                                                "pair_id": [row["pair_id"]],
                                                "label": [row["label"]],
                                                "votes": [votes],
                                                "msg": [msg],
                                                "final_output": [output],
                                            }
                                        ),
                                    ]
                                )

                            case "consider-both":  # consider both
                                msg = [
                                    {
                                        "role": "system",
                                        "content": "You are a legal reasoning system. Given a set of relevant articles, you must answer whether a legal statement is true.",
                                    }
                                ]
                                if args.num_shots > 1:  # if using shots
                                    msg = add_num_shots_to_prompt(
                                        row, msg, num_shots
                                    )  # add shots to prompt
                                msg.append(
                                    {
                                        "role": "user",
                                        "content": f"\n\nArticles:\n{replace_article_ids(row['articles'])}\n\nQuery:\n{row['query']}\n\nCome up with an explanation where the answer is 'N'.",
                                    }
                                )  # append prompt to prompt
                                msg.append(
                                    {
                                        "role": "assistant",
                                        "content": get_output(msg, 1, 1024),
                                    }
                                )  # append model output to prompt
                                msg.append(
                                    {
                                        "role": "user",
                                        "content": "Come up with an explanation where the answer is 'Y'.",
                                    }
                                )
                                msg.append(
                                    {
                                        "role": "assistant",
                                        "content": get_output(msg, 1, 1024),
                                    }
                                )  # append model output to prompt
                                msg.append(
                                    {
                                        "role": "user",
                                        "content": "So is the answer yes (Y) or no (N)? Just answer as 'Y' for yes and 'N' for no enclosed in single quotes. Write nothing else.",
                                    }
                                )  # append final question to prompt
                                if (
                                    args.temperature_sensitive_ensemble
                                ):  # if using temperature-sensitive ensemble
                                    output = get_output(
                                        msg, e / (args.ensemble_size - 1), 1024
                                    )  # get model output
                                else:  # if not using temperature-sensitive ensemble
                                    output = get_output(
                                        msg, args.temperature, 1024
                                    )  # get model output
                                if "'Y'" in output:
                                    votes.append("Y")
                                elif "'N'" in output:
                                    votes.append("N")
                                else:
                                    if len(output.strip()) == 1:
                                        votes.append(output)
                                    else:
                                        votes.append("U")
                                raw_results = pandas.concat(
                                    [
                                        raw_results,
                                        pandas.DataFrame(
                                            {
                                                "pair_id": [row["pair_id"]],
                                                "label": [row["label"]],
                                                "votes": [votes],
                                                "msg": [msg],
                                                "final_output": [output],
                                            }
                                        ),
                                    ]
                                )

                            case "self-ask":  # self-ask
                                msg = [
                                    {
                                        "role": "system",
                                        "content": "You are a legal reasoning system. Given a set of relevant articles, you must answer whether a legal statement is true.",
                                    }
                                ]
                                if args.num_shots > 1:  # if using shots
                                    msg = add_num_shots_to_prompt(
                                        row, msg, num_shots
                                    )  # add shots to prompt
                                msg.append(
                                    {
                                        "role": "user",
                                        "content": f"\n\nArticles:\n{replace_article_ids(row['articles'])}\n\nQuery:\n{row['query']}\n\nHow is the query related to the articles?",
                                    }
                                )  # append prompt to prompt
                                msg.append(
                                    {
                                        "role": "assistant",
                                        "content": get_output(msg, 1, 1024),
                                    }
                                )  # append model output to prompt
                                msg.append(
                                    {
                                        "role": "user",
                                        "content": "What are the steps to find an answer to this query?",
                                    }
                                )
                                msg.append(
                                    {
                                        "role": "assistant",
                                        "content": get_output(msg, 1, 1024),
                                    }
                                )  # append model output to prompt
                                msg.append(
                                    {
                                        "role": "user",
                                        "content": "So is the answer yes (Y) or no (N)? Just answer as 'Y' for yes and 'N' for no enclosed in single quotes. Write nothing else.",
                                    }
                                )  # append final question to prompt
                                if (
                                    args.temperature_sensitive_ensemble
                                ):  # if using temperature-sensitive ensemble
                                    output = get_output(
                                        msg, e / (args.ensemble_size - 1), 1024
                                    )  # get model output
                                else:  # if not using temperature-sensitive ensemble
                                    output = get_output(
                                        msg, args.temperature, 1024
                                    )  # get model output
                                if "'Y'" in output:
                                    votes.append("Y")
                                elif "'N'" in output:
                                    votes.append("N")
                                else:
                                    if len(output.strip()) == 1:
                                        votes.append(output)
                                    else:
                                        votes.append("U")
                                raw_results = pandas.concat(
                                    [
                                        raw_results,
                                        pandas.DataFrame(
                                            {
                                                "pair_id": [row["pair_id"]],
                                                "label": [row["label"]],
                                                "votes": [votes],
                                                "msg": [msg],
                                                "final_output": [output],
                                            }
                                        ),
                                    ]
                                )

                            case "symbolic":  # symbolic
                                # TODO: symbolic
                                pass

                            case _:  # invalid
                                raise ValueError("Invalid prompting style.")
                    prediction = Counter(votes).most_common(1)[0][
                        0
                    ]  # get most common vote
                    if prediction == "U":
                        prediction = random.choice(["Y", "N"])
                    results = pandas.concat(
                        [
                            results,
                            pandas.DataFrame(
                                {
                                    "pair_id": [row["pair_id"]],
                                    "label": [row["label"]],
                                    "votes": [votes],
                                    "prediction": [prediction],
                                }
                            ),
                        ]
                    )
                    pbar.update(1)  # update progress bar
    except:
        print("saving results")
        results.to_csv(results_file, index=False, sep="\t")  # save results
        raw_results.to_csv(raw_results_file, index=False, sep="\t")  # save raw results
        raise
    print(
        colored(
            f"Got {sum(results['label'] == results['prediction'])} right. Accuracy = {sum(results['label'] == results['prediction']) / len(results)}",
            color,
        )
    )
    results.to_csv(results_file, index=False, sep="\t")  # save results
    raw_results.to_csv(raw_results_file, index=False, sep="\t")  # save raw results
