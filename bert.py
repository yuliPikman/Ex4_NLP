import nltk
from tqdm import tqdm
from chu_liu_edmonds import cle_min
from transformer_parser_utils import get_attention_matrix
from perceptron import extract_train_tree, calc_uas
from transformers import AutoTokenizer, AutoModel


# Initialize the pre-trained transformer model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased",
output_attentions=True)
model.eval()


# Download and split the dependency treebank dataset
nltk.download('dependency_treebank')
from nltk.corpus import dependency_treebank
load = dependency_treebank.parsed_sents()
split_idx = int(len(load) * 0.9)
test_corpus = load[split_idx:]


def attn_to_arc_scores(attn_matrix):
    """ Converts a multi-dimensional attention matrix into directional arc scores.

        Note:
            The Chu-Liu-Edmonds algorithm (`cle_min`) calculates the *minimum* spanning tree. Therefore, the attention weights are negated so that
            higher attention values correspond to preferred directed edges.

        Args:
            attn_matrix: A 2D array or PyTorch tensor (shape: [n, n]) representing
              the extracted attention weights between tokens in a sentence.

        Returns:
            A dictionary mapping directed edges `(parent_idx, child_idx)` to their
            negated attention score.
    """
    scores_dict = {}
    n = attn_matrix.shape[0]

    for u in range(n):
        for v in range(1, n):
            if u == v:
                continue
            scores_dict[(u, v)] = -attn_matrix[u,v]

    return scores_dict


def eval_test(layer):
    """ Evaluates the Unlabeled Attachment Score (UAS) of a specific transformer layer.

        Extracts attention matrices from the specified layer for the entire test
        corpus, constructs dependency trees using Chu-Liu-Edmonds, and evaluates
        them against the ground-truth annotations.

        Args:
            layer: The zero-indexed integer specifying which transformer layer's
              attention matrix to extract.

        Returns:
            The average Unlabeled Attachment Score (UAS) for the selected layer across
            the test corpus.
    """
    sum_error = 0

    for t in tqdm(test_corpus, desc="Evaluating Test Set"):
        words = [t.nodes[i]['word'] for i in range(1, len(t.nodes))]
        attn_matrix = get_attention_matrix(words, tokenizer, model, layer)
        scores_dict = attn_to_arc_scores(attn_matrix)
        pred_tree = cle_min(scores_dict, attn_matrix.shape[0])
        gt_tree = extract_train_tree(t.nodes)
        sum_error = calc_uas(gt_tree, pred_tree, sum_error, t)

    avg = sum_error / len(test_corpus)
    return avg


if __name__ == '__main__':
    layers = [0, 5,11]
    for curr_layer in layers:
        print("uas of layer", curr_layer, ":", eval_test(curr_layer))
