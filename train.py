import re
import math
import random
from collections import Counter
from typing import Generator, List, Tuple, Dict

import numpy as np

def load_text8(path, max_tokens=None):
    with open(path, "r", encoding="utf-8") as f:
        words = f.read().split()
    if max_tokens:
        words = words[:max_tokens]
    return " ".join(words)



class Word2VecSGNS:
    def __init__(
        self,
        embedding_dim: int = 50,
        window_size: int = 2,
        num_negative: int = 5,
        learning_rate: float = 0.025,
        min_count: int = 2,
        unigram_power: float = 0.75,
        subsample_t: float = 1e-2,
        seed: int = 42,
    ):
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.num_negative = num_negative
        self.learning_rate = learning_rate
        self.min_count = min_count
        self.unigram_power = unigram_power
        self.subsample_t = subsample_t
        self.seed = seed

        random.seed(seed)
        np.random.seed(seed)

        self.word_to_idx: Dict[str, int] = {}
        self.idx_to_word: List[str] = []
        self.word_counts: Counter = Counter()

        self.vocab_size = 0

        # Input and output embeddings
        self.W_in = None   # shape: (V, D)
        self.W_out = None  # shape: (V, D)

        # Negative sampling distribution
        self.neg_sampling_probs = None

    @staticmethod
    def tokenize(text: str) -> List[str]:
        text = text.lower()
        tokens = re.findall(r"[a-z']+", text)
        return tokens

    def build_vocab(self, tokens: List[str]) -> List[int]:
        """
        Builds vocabulary from tokens and returns corpus encoded as indices.
        Words with frequency < min_count are discarded.
        """
        self.word_counts = Counter(tokens)

        filtered_words = [word for word, count in self.word_counts.items() if count >= self.min_count]
        filtered_words.sort()

        self.idx_to_word = filtered_words
        self.word_to_idx = {word: idx for idx, word in enumerate(self.idx_to_word)}
        self.vocab_size = len(self.idx_to_word)

        corpus_indices = [self.word_to_idx[word] for word in tokens if word in self.word_to_idx]

        return corpus_indices

    def subsample_corpus(self, corpus_indices: List[int]) -> List[int]:
        """
        Subsample frequent words following Mikolov et al. (2013):

            P(keep | w) = min(1, sqrt(t / f(w)) + t / f(w))

        where f(w) is the relative frequency of word w and t is a threshold
        (typically 1e-4). Common words like "the", "a", "is" are discarded with
        high probability, which speeds up training and improves embedding quality
        for rarer, more informative words.
        """
        total = len(corpus_indices)
        kept = []
        for idx in corpus_indices:
            freq = self.word_counts[self.idx_to_word[idx]] / total
            keep_prob = min(1.0, (math.sqrt(freq / self.subsample_t) + 1) * (self.subsample_t / freq))
            if random.random() < keep_prob:
                kept.append(idx)
        return kept

    def init_parameters(self):
        """
        Initializes embeddings.
        Standard choice:
        - input embeddings small random values
        - output embeddings zeros or small random values
        """
        limit = 0.5 / self.embedding_dim
        self.W_in = np.random.uniform(
            low=-limit,
            high=limit,
            size=(self.vocab_size, self.embedding_dim)
        ).astype(np.float64)

        self.W_out = np.random.uniform(
            low=-limit,
            high=limit,
            size=(self.vocab_size, self.embedding_dim)
        ).astype(np.float64)

    def build_negative_sampling_distribution(self):
        """
        Build unigram distribution raised to the 0.75 power:
            P(w) proportional to count(w)^0.75
        """
        counts = np.array(
            [self.word_counts[word] for word in self.idx_to_word],
            dtype=np.float64
        )
        adjusted = counts ** self.unigram_power
        self.neg_sampling_probs = adjusted / adjusted.sum()

    def iter_training_pairs(
        self, corpus_indices: List[int]
    ) -> Generator[Tuple[int, int], None, None]:
        """
        Generator that yields (center, context) pairs for skip-gram.

        Pairs are produced on the fly — no full list is built in memory.
        The corpus_indices array (one int per token) is the only structure
        kept in RAM, which is orders of magnitude smaller than the pairs list.

        To shuffle across epochs, callers should shuffle corpus_indices before
        passing it here (see fit()).
        """
        n = len(corpus_indices)
        for center_pos in range(n):
            center_word = corpus_indices[center_pos]
            left  = max(0, center_pos - self.window_size)
            right = min(n, center_pos + self.window_size + 1)
            for context_pos in range(left, right):
                if context_pos != center_pos:
                    yield center_word, corpus_indices[context_pos]

    def sample_negative_indices(self, positive_idx: int) -> np.ndarray:
        """
        Sample negative words according to the negative sampling distribution.
        Avoid the true positive context word.
        """
        negatives = []
        while len(negatives) < self.num_negative:
            sampled = np.random.choice(self.vocab_size, p=self.neg_sampling_probs)
            if sampled != positive_idx:
                negatives.append(sampled)
        return np.array(negatives, dtype=np.int64)

    @staticmethod
    def sigmoid(x):
        """
        Numerically stable sigmoid.
        """
        x = np.clip(x, -15, 15)
        return 1.0 / (1.0 + np.exp(-x))

    def train_pair(self, center_idx: int, context_idx: int) -> float:
        """
        Train on one positive (center, context) pair using SGNS.

        Loss for one pair:
            L = -log(sigmoid(u_o^T v_c)) - sum_i log(sigmoid(-u_ni^T v_c))

        where:
            v_c  = input embedding of center word
            u_o  = output embedding of true context word
            u_ni = output embedding of negative sample i

        Note on views vs copies:
            v_c and u_o are views into W_in/W_out via basic indexing.
            U_neg is already a copy because NumPy fancy indexing always copies.
            All gradients are fully computed before any in-place update, so
            no explicit .copy() calls are needed — the views are safe to use.
        """
        v_c = self.W_in[center_idx]        # view, shape (D,)
        u_o = self.W_out[context_idx]      # view, shape (D,)

        negative_indices = self.sample_negative_indices(context_idx)
        U_neg = self.W_out[negative_indices]   # copy via fancy indexing, shape (K, D)

        # ---------- Forward pass ----------
        pos_score = np.dot(u_o, v_c)                 # scalar
        neg_scores = np.dot(U_neg, v_c)              # shape (K,)

        pos_sigmoid = self.sigmoid(pos_score)        # sigma(u_o^T v_c)
        neg_sigmoid = self.sigmoid(neg_scores)       # sigma(u_n^T v_c)

        # SGNS loss
        loss_pos = -np.log(pos_sigmoid + 1e-10)
        loss_neg = -np.sum(np.log(1.0 - neg_sigmoid + 1e-10))
        loss = loss_pos + loss_neg

        # ---------- Gradients ----------
        # Positive term:
        # d/dx[-log(sigmoid(x))] = sigmoid(x) - 1
        grad_pos = pos_sigmoid - 1.0  # scalar

        # Negative terms:
        # d/dx[-log(sigmoid(-x))] = sigmoid(x)
        grad_neg = neg_sigmoid        # shape (K,)

        # Gradient wrt center embedding v_c
        grad_v = grad_pos * u_o + np.sum(grad_neg[:, None] * U_neg, axis=0)

        # Gradient wrt positive output embedding u_o
        grad_u_o = grad_pos * v_c

        # Gradient wrt negative output embeddings U_neg
        grad_U_neg = grad_neg[:, None] * v_c[None, :]

        # ---------- Parameter updates ----------
        self.W_in[center_idx]  -= self.learning_rate * grad_v
        self.W_out[context_idx] -= self.learning_rate * grad_u_o

        for i, neg_idx in enumerate(negative_indices):
            self.W_out[neg_idx] -= self.learning_rate * grad_U_neg[i]

        return float(loss)

    def fit(
        self,
        text: str,
        epochs: int = 5,
        verbose: bool = True,
    ):
        """
        Full training pipeline:
        1. tokenize
        2. build vocab
        3. initialize parameters
        4. build negative sampling distribution
        5. per epoch:
             a. re-subsample corpus (stochastic, preserves original order)
             b. materialize pairs from the subsampled corpus
             c. shuffle the pairs list
             d. train

        Why materialize + shuffle instead of a generator?
            The context window must always slide over the original token sequence —
            reordering tokens would invent co-occurrences that never existed in the
            text. The correct way to shuffle training signal is to shuffle the
            already-generated pairs, not the corpus. For typical corpora (< 1M
            tokens, window=2) the pairs list fits comfortably in RAM (~40–80 MB).
            If RAM is a hard constraint, replace with the iter_training_pairs
            generator and accept a fixed presentation order each epoch.

        Why re-subsample each epoch?
            subsample_corpus is stochastic: each token is kept with some
            probability. Running it again each epoch produces a different
            subset of tokens, so the model sees a varied training signal
            across epochs even before the pairs shuffle.
        """
        tokens = self.tokenize(text)
        base_corpus = self.build_vocab(tokens)

        if len(base_corpus) == 0:
            raise ValueError("Corpus is empty after vocabulary filtering. Lower min_count or provide more text.")

        self.init_parameters()
        self.build_negative_sampling_distribution()

        if verbose:
            print(f"Number of raw tokens       : {len(tokens)}")
            print(f"Vocabulary size            : {self.vocab_size}")

        for epoch in range(epochs):
            # Fresh stochastic subsample each epoch (order is preserved).
            corpus_indices = self.subsample_corpus(base_corpus)
            if len(corpus_indices) == 0:
                raise ValueError("Corpus is empty after subsampling. Lower subsample_t or provide more text.")

            # Materialize all pairs for this epoch, then shuffle.
            pairs = list(self.iter_training_pairs(corpus_indices))
            random.shuffle(pairs)

            if verbose and epoch == 0:
                print(f"Tokens after subsampling   : {len(corpus_indices)} (varies per epoch)")
                print(f"Training pairs (epoch 1)   : {len(pairs)}")

            total_loss = 0.0
            total_pairs = len(pairs)

            for step, (center_idx, context_idx) in enumerate(pairs, start=1):
                loss = self.train_pair(center_idx, context_idx)
                total_loss += loss

                if verbose and step % 10000 == 0:
                    avg_loss = total_loss / step
                    print(f"Epoch {epoch + 1}/{epochs}, step {step}/{total_pairs}, avg_loss={avg_loss:.4f}")

            avg_epoch_loss = total_loss / total_pairs
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs} finished. Average loss: {avg_epoch_loss:.4f}")

    def get_embedding(self, word: str) -> np.ndarray:
        if word not in self.word_to_idx:
            raise ValueError(f"Word '{word}' not in vocabulary.")
        idx = self.word_to_idx[word]
        return self.W_in[idx]

    def most_similar(self, word: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Find nearest neighbors using cosine similarity on input embeddings.
        """
        if word not in self.word_to_idx:
            raise ValueError(f"Word '{word}' not in vocabulary.")

        query_idx = self.word_to_idx[word]
        query_vec = self.W_in[query_idx]

        query_norm = np.linalg.norm(query_vec) + 1e-10
        all_norms  = np.linalg.norm(self.W_in, axis=1) + 1e-10

        sims = np.dot(self.W_in, query_vec) / (all_norms * query_norm)

        best_indices = np.argsort(-sims)

        results = []
        for idx in best_indices:
            if idx == query_idx:
                continue
            results.append((self.idx_to_word[idx], float(sims[idx])))
            if len(results) == top_k:
                break

        return results

    def analogy(
        self, pos1: str, neg1: str, pos2: str, top_k: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Solve a 3CosAdd analogy:  pos1 - neg1 + pos2 ≈ ?

        Classic example:  king - man + woman ≈ queen

        The query vector is formed as:
            q = v(pos1) - v(neg1) + v(pos2)

        then ranked by cosine similarity against all vocabulary vectors.
        The three input words are excluded from results.
        """
        for w in [pos1, neg1, pos2]:
            if w not in self.word_to_idx:
                raise ValueError(f"Word '{w}' not in vocabulary.")

        query_vec = (
            self.get_embedding(pos1)
            - self.get_embedding(neg1)
            + self.get_embedding(pos2)
        )

        query_norm = np.linalg.norm(query_vec) + 1e-10
        all_norms  = np.linalg.norm(self.W_in, axis=1) + 1e-10
        sims = np.dot(self.W_in, query_vec) / (all_norms * query_norm)

        exclude = {self.word_to_idx[w] for w in [pos1, neg1, pos2]}
        best_indices = np.argsort(-sims)

        results = []
        for idx in best_indices:
            if idx in exclude:
                continue
            results.append((self.idx_to_word[idx], float(sims[idx])))
            if len(results) == top_k:
                break

        return results


if __name__ == "__main__":
    USE_TEXT8  = False
    MAX_TOKENS = 1_000_000   # ignored when USE_TEXT8 = False

    if USE_TEXT8:
        text = load_text8("data/text8", max_tokens=MAX_TOKENS)
        params = dict(embedding_dim=100, window_size=5, num_negative=5,
                      learning_rate=0.025, min_count=5, subsample_t=1e-4)
        test_words = ["king", "queen", "france", "paris", "computer", "science"]
        analogy_tests = [
            ("paris",  "france", "italy"),
            ("king",   "man",    "woman"),
            ("actor",  "man",    "woman"),
        ]
    else:
        with open("data/holmes.txt", "r", encoding="utf-8") as f:
            text = f.read()
        params = dict(embedding_dim=50, window_size=5, num_negative=5,
                      learning_rate=0.025, min_count=3, subsample_t=1e-3)
        test_words = [
            "holmes",
            "watson",
            "sherlock",
            "case",
            "detective",
            "police",
            "london",
            "house",
        ]
        analogy_tests = [
            ("holmes", "detective", "watson"),
            ("london", "england", "paris"),
            ("man", "woman", "sir"),
        ]

    model = Word2VecSGNS(**params, seed=42)
    model.fit(text, epochs=30, verbose=True)

    for word in test_words:
        if word in model.word_to_idx:
            print(f"\nMost similar to '{word}':")
            for neighbor, score in model.most_similar(word, top_k=5):
                print(f"  {neighbor:15s} {score:.4f}")

    print("\n--- Analogies ---")
    for pos1, neg1, pos2 in analogy_tests:
        if all(w in model.word_to_idx for w in [pos1, neg1, pos2]):
            print(f"  {pos1} - {neg1} + {pos2} =")
            for w, s in model.analogy(pos1, neg1, pos2, top_k=3):
                print(f"    {w:15s} {s:.4f}")
        else:
            print(f"  Skipping: {pos1} - {neg1} + {pos2} (word not in vocab)")