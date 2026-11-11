from collections import Counter
import numpy as np
import pandas as pd
import os
import pyarrow.parquet as pq
import scipy.sparse as sp
from scipy.special import xlogy
from sklearn.feature_extraction.text import CountVectorizer

TOKEN_PATTERN = r"(?u)\b[^\W\d_]{2,}\b"

def iter_corpus_chunks(file_path, chunk_size=50000, cols=("abstract", "label")):
    """Streams chunks from either a .parquet or .csv file without loading all into RAM."""
    if file_path.endswith(".parquet"):
        pf = pq.ParquetFile(file_path)
        for batch in pf.iter_batches(batch_size=chunk_size, columns=list(cols)):
            yield batch.to_pandas()
    else:
        for chunk in pd.read_csv(file_path, usecols=list(cols), chunksize=chunk_size):
            yield chunk

FILE_PATH = os.path.join('..', 'query_data', 'reproduction_papers.parquet')
CHUNK_SIZE = 50000
GLOBAL_MIN_DF = 50   # Minimum total documents across the 400k corpus
CHUNK_MIN_DF = 3     # Drops 1-off typos inside each 50k chunk to save RAM

counts_1 = Counter()
counts_0 = Counter()
total_pos = 0
total_neg = 0

print("Pass 1: Streaming chunks to accumulate global 1-gram, 2-gram, 3-gram counts...")
for i, chunk in enumerate(iter_corpus_chunks(FILE_PATH, CHUNK_SIZE)):
    texts = chunk["abstract"].fillna("")
    y_chunk = chunk["label"].astype(int).to_numpy()
    
    pos_mask = (y_chunk == 1)
    neg_mask = (y_chunk == 0)
    total_pos += int(pos_mask.sum())
    total_neg += int(neg_mask.sum())

    vec = CountVectorizer(
        ngram_range=(1, 3),
        binary=True,
        min_df=CHUNK_MIN_DF,
        lowercase=True,
        token_pattern=TOKEN_PATTERN
    )
    X_chunk = vec.fit_transform(texts)
    vocab_chunk = vec.get_feature_names_out()

    c1 = np.asarray(X_chunk[pos_mask].sum(axis=0)).ravel()
    c0 = np.asarray(X_chunk[neg_mask].sum(axis=0)).ravel()

    # Accumulate non-zero document counts into global counters
    for term, n1 in zip(vocab_chunk[c1 > 0], c1[c1 > 0]):
        counts_1[term] += int(n1)
    for term, n0 in zip(vocab_chunk[c0 > 0], c0[c0 > 0]):
        counts_0[term] += int(n0)

    print(f"  Processed chunk {i+1} (running total: {total_pos + total_neg:,} docs)")

# Filter to n-grams meeting GLOBAL_MIN_DF across all chunks
all_terms = set(counts_1.keys()) | set(counts_0.keys())
vocab = np.array([t for t in all_terms if (counts_1[t] + counts_0[t]) >= GLOBAL_MIN_DF])

y_T = np.array([counts_1[t] for t in vocab], dtype=np.float64)
y_ST = np.array([counts_0[t] for t in vocab], dtype=np.float64)

# Compute Supervisor's Table Percentages & KLR Log-Likelihood Score
pct_1 = (y_T / total_pos) * 100.0
pct_0 = (y_ST / total_neg) * 100.0
lift = (pct_1 + 0.01) / (pct_0 + 0.01)

p_T = (y_T + 1.0) / (total_pos + 2.0)
p_ST = (y_ST + 1.0) / (total_neg + 2.0)
p_pool = (y_T + y_ST + 2.0) / (total_pos + total_neg + 4.0)

ll_T = xlogy(y_T + 1.0, p_T / p_pool) + xlogy(total_pos - y_T + 1.0, (1.0 - p_T) / (1.0 - p_pool))
ll_ST = xlogy(y_ST + 1.0, p_ST / p_pool) + xlogy(total_neg - y_ST + 1.0, (1.0 - p_ST) / (1.0 - p_pool))
klr_score = 2.0 * (ll_T + ll_ST)

ngram_table = pd.DataFrame({
    "gram_type": [f"{w.count(' ') + 1}-grams" for w in vocab],
    "ngram": vocab,
    "pct_1": np.round(pct_1, 2),
    "pct_0": np.round(pct_0, 2),
    "count_1": y_T.astype(int),
    "count_0": y_ST.astype(int),
    "net_diff_pct": np.round(pct_1 - pct_0, 2),
    "lift_ratio": np.round(lift, 2),
    "klr_likelihood": np.round(klr_score, 2)
}).sort_values(by="klr_likelihood", ascending=False)

OUTPATH_NGRAM = os.path.join('results', "ngram_distribution_table.csv")
OUTPATH_GREEDY = os.path.join('results', "optimal_greedy_filters.csv")
ngram_table.to_csv(OUTPATH_NGRAM, index=False)
print(f"Saved {len(ngram_table):,} global n-grams to ngram_distribution_table.csv")

# -------------------------------------------------------------------------
# PASS 2: Fixed-Vocabulary Chunking for Greedy Maximum Coverage
# -------------------------------------------------------------------------
# Lock vocabulary to the top 10,000 positive candidates so the 400k matrix stays <150MB
MAX_CANDIDATES = 10000
MIN_LIFT = 3.0

cand_df = ngram_table[(ngram_table["pct_1"] > ngram_table["pct_0"]) & (ngram_table["lift_ratio"] >= MIN_LIFT)].head(MAX_CANDIDATES)
cand_vocab = cand_df["ngram"].to_numpy()

fixed_vec = CountVectorizer(
    ngram_range=(1, 3),
    vocabulary=cand_vocab,
    binary=True,
    lowercase=True,
    token_pattern=TOKEN_PATTERN
)

print(f"Pass 2: Building lightweight sparse matrix for top {len(cand_vocab):,} candidates...")
sparse_blocks = []
y_blocks = []

for chunk in iter_corpus_chunks(FILE_PATH, CHUNK_SIZE):
    sparse_blocks.append(fixed_vec.transform(chunk["abstract"].fillna("")))
    y_blocks.append(chunk["label"].astype(int).to_numpy())

X_cand = sp.vstack(sparse_blocks, format="csc")
y_all = np.concatenate(y_blocks)

# Run Greedy Maximum Coverage on the global CSC matrix
pos_mask_all = (y_all == 1)
neg_mask_all = (y_all == 0)
uncovered_pos = pos_mask_all.copy()
covered_any = np.zeros(len(y_all), dtype=bool)

selected_filters = []
for step in range(1, 201):
    new_pos_gains = np.asarray(X_cand[uncovered_pos, :].sum(axis=0)).ravel()
    if new_pos_gains.max() == 0:
        break

    uncovered_neg = neg_mask_all & (~covered_any)
    new_neg_costs = np.asarray(X_cand[uncovered_neg, :].sum(axis=0)).ravel()

    marginal_score = new_pos_gains / (new_neg_costs + 10.0)
    best_idx = int(np.argmax(marginal_score))

    matching_docs = X_cand.indices[X_cand.indptr[best_idx]:X_cand.indptr[best_idx + 1]]
    covered_any[matching_docs] = True
    uncovered_pos[matching_docs] = False

    cum_recall_1 = (covered_any & pos_mask_all).sum() / total_pos
    cum_fpr_0 = (covered_any & neg_mask_all).sum() / total_neg

    selected_filters.append({
        "step": step,
        "ngram": cand_vocab[best_idx],
        "new_1s_added": int(new_pos_gains[best_idx]),
        "cumulative_pct_1_captured": round(cum_recall_1 * 100, 2),
        "cumulative_pct_0_kept": round(cum_fpr_0 * 100, 2)
    })
    if cum_recall_1 >= 0.95:
        break
pd.DataFrame(selected_filters).to_csv(OUTPATH_GREEDY, index=False)