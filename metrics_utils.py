import nltk
import nltk.translate.bleu_score as bleu
from nltk import word_tokenize
import numpy as np

from typing import List, Dict

try:
  nltk.data.find('tokenizers/punkt')
except LookupError:
  nltk.download('punkt')


def calculate_metrics(predicted_texts: List[str], references: List[List[str]]) -> Dict[str, float]:

    # splitting texts into words
    predicted_tokenized_texts = [word_tokenize(text) for text in predicted_texts]
    references_tokenized = [
        [word_tokenize(reference_text) for reference_text in reference_examples] for reference_examples in references
    ]

    sentence_bleu_scores = [
        bleu.sentence_bleu(reference, candidate) for reference, candidate in zip(references_tokenized, predicted_tokenized_texts)
    ]
    corpus_bleu_score = bleu.corpus_bleu(references, predicted_tokenized_texts)

    two_grams_sentence_bleu_scores = [
        bleu.sentence_bleu(reference, candidate, 2) for reference, candidate in zip(references_tokenized, predicted_tokenized_texts)
    ]
    three_grams_sentence_bleu_scores = [
        bleu.sentence_bleu(reference, candidate, 3) for reference, candidate in
        zip(references_tokenized, predicted_tokenized_texts)
    ]

    return {
        "median_sentence_bleu_scores": np.median(sentence_bleu_scores).item(),
        "mean_sentence_bleu_scores": np.mean(sentence_bleu_scores).item(),
        "var_sentence_bleu_scores": np.var(sentence_bleu_scores).item(),
        "corpus_bleu_score": corpus_bleu_score,

        "median_two_grams_sentence_bleu_scores": np.median(two_grams_sentence_bleu_scores).item(),
        "mean_two_grams_sentence_bleu_scores": np.mean(two_grams_sentence_bleu_scores).item(),
        "var_two_grams_sentence_bleu_scores": np.var(two_grams_sentence_bleu_scores).item(),

        "median_three_grams_sentence_bleu_scores": np.median(three_grams_sentence_bleu_scores).item(),
        "mean_three_grams_sentence_bleu_scores": np.mean(three_grams_sentence_bleu_scores).item(),
        "var_three_grams_sentence_bleu_scores": np.var(three_grams_sentence_bleu_scores).item(),
    }





