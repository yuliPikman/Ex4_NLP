import random
from typing import Any
import nltk
from chu_liu_edmonds import cle_min
from tqdm import tqdm


# Download and split the dependency treebank dataset
nltk.download('dependency_treebank')
from nltk.corpus import dependency_treebank
load = dependency_treebank.parsed_sents()
split_idx = int(len(load) * 0.9)
train_corpus = load[:split_idx]
test_corpus = load[split_idx:]


def feature_func(edge, nodes):
    """ Extracts features for a given directed edge between two nodes.

        Args:
            edge: A tuple representing the directed edge (parent_idx, child_idx).
            nodes: A list of dictionaries, where each dictionary represents a node
              containing 'word' and 'tag' (POS tag) fields.

        Returns:
            A list of tuples, where each tuple represents a specific feature
            (e.g., bigram or POS tag combination) for the edge.
    """
    v = nodes[edge[1]]
    u = nodes[edge[0]]
    word_feature = ("bigram", u["word"], v["word"])
    pos_feature = ("POS", u["tag"], v["tag"])

    return [word_feature, pos_feature]


def score(w, nodes, n):
    """ Calculates the edge weights for all possible pairs of nodes based on current feature weights.

        Note:
            The Chu-Liu-Edmonds algorithm used (`cle_min`) finds the *minimum* spanning tree,
            so the scores returned are negated to allow standard maximum-weight optimization.

        Args:
            w: A dictionary mapping features to their learned weights.
            nodes: A list of dictionaries representing the sentence tokens.
            n: The total number of nodes in the sentence (including ROOT).

        Returns:
            A dictionary mapping directed edges '(u, v)' to their negated score.
    """
    score_dict = {}

    for u in range(n):
        for v in range(1, n):
            if u == v:
                continue

            features = feature_func((u, v), nodes)
            edge_score = sum(w.get(f, 0) for f in features)
            score_dict[(u, v)] = -edge_score

    return score_dict


def extract_train_tree(nodes):
    """ Extracts the ground-truth dependency tree from the annotated sentence nodes.

       Args:
           nodes: A list of dictionaries representing tokens, where each token
             contains a 'head' key pointing to its parent index.

       Returns:
           A dictionary mapping child node indices to their respective head node indices.
    """
    train_tree = {}

    for i in range(1, len(nodes)):
        head = nodes[i]['head']
        if head is not None:
            train_tree[i] = head

    return train_tree


def update_weight_by_tree(w, t, nodes, sign=1):
    """ Updates the global feature weights based on the structure of a given tree.

        Args:
            w: The dictionary containing the current feature weights.
            t: A dependency tree representation mapping child indices to parent indices.
            nodes: A list of dictionaries representing sentence tokens.
            sign: The step direction for the weight update. Use `1` to reward features
              (ground-truth) and `-1` to penalize features (incorrect predictions).

        Returns:
            The updated weight dictionary.
    """
    for edge in t.items():
        v, u = edge
        key_bigram = ("bigram", nodes[u]["word"], nodes[v]["word"])
        key_pos = ("POS", nodes[u]["tag"], nodes[v]["tag"])
        w[key_bigram] = w.get(key_bigram, 0) + sign
        w[key_pos] = w.get(key_pos, 0) + sign

    return w


def perceptron_helper(sentence, w):
    """ Preprocesses a sentence, predicts its tree structure, and extracts the ground truth.

        Args:
            sentence: A parsed sentence object from the NLTK dependency treebank.
            w: The dictionary containing current feature weights.

        Returns:
            A tuple containing:
                - pred_tree: The predicted dependency tree dictionary.
                - gt_tree: The ground-truth dependency tree dictionary.
    """
    n = len(sentence.nodes)
    sentence.nodes[0]["word"] = "ROOT"
    sentence.nodes[0]["tag"] = "ROOT"
    score_dict = score(w, sentence.nodes, n)
    pred_tree = cle_min(score_dict, n)
    gt_tree = extract_train_tree(sentence.nodes)

    return pred_tree, gt_tree


def perceptron(train):
    """ Trains a structured perceptron model over the training corpus to learn dependency features.

        Trains for 2 epochs, shuffling the dataset at the start of each epoch, and averages
        the accumulated weights at the end.

        Args:
            train: A list of training parsed sentence objects.

        Returns:
            A dictionary containing the averaged, learned feature weights.
    """
    w = {}
    w_sum = {}

    for i in range(2):
        random.shuffle(train)

        for sentence in tqdm(train, desc=f"Training Epoch {i+1}/2"):
            pred_tree, gt_tree = perceptron_helper(sentence, w)
            w = update_weight_by_tree(w, pred_tree, sentence.nodes, -1)
            w = update_weight_by_tree(w, gt_tree, sentence.nodes)

            for f in w:
                w_sum[f] = w_sum.get(f, 0) + w[f]

    for f in w_sum:
        w_sum[f] /= (2 * len(train))

    return w_sum


def evaluate_perceptron(test, train):
    """ Trains the perceptron on the training set and evaluates its performance on the test set.

            Args:
                test: A list of test parsed sentence objects.
                train: A list of training parsed sentence objects.

            Returns:
                The average Unlabeled Attachment Score (UAS) across the entire test set.
    """
    w_prime = perceptron(train)
    sum_correct = 0

    for t in tqdm(test, desc="Evaluating Test Set"):
        pred_tree, gt_tree = perceptron_helper(t, w_prime)
        sum_correct = calc_uas(gt_tree, pred_tree, sum_correct, t)

    avg = sum_correct / len(test)
    return avg


def calc_uas(gt_tree: dict[Any, Any], pred_tree: dict[Any, Any], sum_correct: int, t) -> int:
    """ Calculates and accumulates the Unlabeled Attachment Score (UAS) for a single sentence.

       Args:
           gt_tree: The ground-truth tree mapping child nodes to parent nodes.
           pred_tree: The predicted tree mapping child nodes to parent nodes.
           sum_correct: The running total of accumulated UAS scores across sentences.
           t: The parsed sentence object (used to determine token length).

       Returns:
           The updated running total of the accumulated UAS score.
    """
    correct_edges = 0

    for child, parent in gt_tree.items():
        if pred_tree.get(child) == parent:
            correct_edges += 1

    sum_correct += correct_edges / (len(t.nodes) - 1)
    return sum_correct


if __name__ == '__main__':
    print("attachment score (UAS) for the learned w", evaluate_perceptron(test_corpus, train_corpus))

