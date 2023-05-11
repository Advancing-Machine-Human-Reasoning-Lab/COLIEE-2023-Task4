from sentence_transformers import SentenceTransformer, util
from bleurt.score import BleurtScorer
from bert_score import BERTScorer
import os, numpy, pandas


def get_similarity(
    model_name, sentences1, sentences2, return_most_similar=0, return_most_dissimilar=0
):
    """
    Returns a list of similarities between sentences1 and sentences2.
    model_name: Name of the model to use.
    sentences1: List of sentences. Can be the train set.
    sentences2: List of sentences. Can be the test example.
    return_most_similar: If > 0, returns the most similar sentences.
    return_most_dissimilar: If > 0, returns the most dissimilar sentences.
    """
    sts_scores = []
    match model_name:
        case "sbert":
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings1 = model.encode(sentences1, convert_to_tensor=True)
            embeddings2 = model.encode(sentences2, convert_to_tensor=True)
            sts_scores = [
                score[0]
                for score in util.cos_sim(embeddings1, embeddings2).cpu().numpy()
            ]
        case "bleurt":
            if os.path.exists("/home/animesh/storage/COLIEE-2023-Task-4/bleurt.tsv"):
                bleurt_df = pandas.read_csv(
                    "/home/animesh/storage/COLIEE-2023-Task-4/bleurt.tsv", sep="\t"
                )
            else:
                bleurt_df = pandas.DataFrame(
                    columns=["sentence1", "sentence2", "score"]
                )
            for s1 in sentences1:
                if (
                    s1 in bleurt_df["sentence1"].values
                    and sentences2[0] in bleurt_df["sentence2"].values
                ):
                    sts_scores.append(
                        bleurt_df[
                            (bleurt_df["sentence1"] == s1)
                            & (bleurt_df["sentence2"] == sentences2[0])
                        ]["score"].values[0]
                    )
                else:
                    bleurt_scorer = BleurtScorer(
                        "/home/animesh/storage/COLIEE-2023-Task-4/bleurt/BLEURT-20/"
                    )
                    score = bleurt_scorer.score(references=[s1], candidates=sentences2)[
                        0
                    ]
                    bleurt_df = pandas.concat(
                        [
                            bleurt_df,
                            pandas.DataFrame(
                                {
                                    "sentence1": [s1],
                                    "sentence2": sentences2,
                                    "score": [score],
                                }
                            ),
                        ]
                    )
                    sts_scores.append(score)
            bleurt_df.to_csv(
                "/home/animesh/storage/COLIEE-2023-Task-4/bleurt.tsv",
                sep="\t",
                index=False,
            )
        case "bert-score":
            if os.path.exists(
                "/home/animesh/storage/COLIEE-2023-Task-4/bert_score.tsv"
            ):
                bert_score_df = pandas.read_csv(
                    "/home/animesh/storage/COLIEE-2023-Task-4/bert_score.tsv", sep="\t"
                )
            else:
                bert_score_df = pandas.DataFrame(
                    columns=["sentence1", "sentence2", "score"]
                )
            for s1 in sentences1:
                if (
                    s1 in bert_score_df["sentence1"].values
                    and sentences2[0] in bert_score_df["sentence2"].values
                ):
                    sts_scores.append(
                        bert_score_df[
                            (bert_score_df["sentence1"] == s1)
                            & (bert_score_df["sentence2"] == sentences2[0])
                        ]["score"].values[0]
                    )
                else:
                    scorer = BERTScorer(lang="en", rescale_with_baseline=True)
                    score = scorer.score([s1], sentences2)[0].mean().item()
                    bert_score_df = pandas.concat(
                        [
                            bert_score_df,
                            pandas.DataFrame(
                                {
                                    "sentence1": [s1],
                                    "sentence2": sentences2,
                                    "score": [score],
                                }
                            ),
                        ]
                    )
                    sts_scores.append(score)
            bert_score_df.to_csv(
                "/home/animesh/storage/COLIEE-2023-Task-4/bert_score.tsv",
                sep="\t",
                index=False,
            )
    match return_most_similar, return_most_dissimilar:
        case 0, 0:
            return
        case 0, 1:
            return [numpy.argmin(sts_scores)]
        case 0, _:
            numpy.argsort(sts_scores)[:return_most_dissimilar]
        case 1, 0:
            return [numpy.argmax(sts_scores)]
        case 1, 1:
            return [numpy.argmax(sts_scores), numpy.argmin(sts_scores)]
        case 1, _:
            return [numpy.argmax(sts_scores)] + numpy.argsort(sts_scores)[
                :return_most_dissimilar
            ]
        case _, 0:
            return numpy.argsort(sts_scores)[-return_most_similar:]
        case _, 1:
            return numpy.argsort(sts_scores)[-return_most_similar:] + [
                numpy.argmin(sts_scores)
            ]
        case _, _:
            return (
                numpy.argsort(sts_scores)[-return_most_similar:]
                + numpy.argsort(sts_scores)[:return_most_dissimilar]
            )
