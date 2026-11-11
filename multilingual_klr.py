import numpy as np
import pandas as pd
from scipy.special import xlogy
from sklearn.feature_extraction.text import CountVectorizer

def run_multilingual_klr(
    df,
    text_col="abstract",
    label_col="label",
    lang_col="language",
    min_docs_per_lang=200,
    min_df_ratio=0.001,  # 0.1% of documents in that language
    abs_min_df=10
):
    """
    Runs KLR (2017) Table 1 Step 3 stratified by OpenAlex 'language' column,
    then combines top candidates for global Greedy Maximum Coverage.
    """
    df = df.copy()
    df[text_col] = df[text_col].fillna("")
    df[lang_col] = df[lang_col].fillna("unknown")
    y_all = df[label_col].astype(int).to_numpy()

    total_pos = int((y_all == 1).sum())
    total_neg = int((y_all == 0).sum())

    all_tables_T = []
    all_tables_ST = []

    # Process each language with at least min_docs_per_lang papers
    lang_counts = df[lang_col].value_counts()
    valid_langs = lang_counts[lang_counts >= min_docs_per_lang].index.tolist()

    for lang in valid_langs:
        sub = df[df[lang_col] == lang]
        y_lang = sub[label_col].astype(int).to_numpy()
        pos_mask = (y_lang == 1)
        neg_mask = (y_lang == 0)

        n_T = int(pos_mask.sum())
        n_ST = int(neg_mask.sum())
        if n_T < 20 or n_ST < 20:
            continue

        # Step 3(a): Language-proportional frequency threshold K_S
        min_df_lang = max(abs_min_df, int(len(sub) * min_df_ratio))

        # For CJK languages (zh, ja, ko), switch to character n-grams (2 to 4 chars)
        if lang in {"zh", "ja", "ko"}:
            vectorizer = CountVectorizer(
                analyzer="char",
                ngram_range=(2, 4),
                binary=True,
                min_df=min_df_lang
            )
        else:
            vectorizer = CountVectorizer(
                ngram_range=(1, 3),
                binary=True,
                min_df=min_df_lang,
                lowercase=True,
                token_pattern=r"(?u)\b[^\W\d_]{2,}\b"
            )

        try:
            X_lang = vectorizer.fit_transform(sub[text_col])
        except ValueError:
            continue

        vocab = np.array(vectorizer.get_feature_names_out())

        # Step 3(b): Document counts & percentages within this language
        y_T = np.asarray(X_lang[pos_mask].sum(axis=0)).ravel()
        y_ST = np.asarray(X_lang[neg_mask].sum(axis=0)).ravel()

        pct_1_lang = (y_T / n_T) * 100.0
        pct_0_lang = (y_ST / n_ST) * 100.0

        # Global percentages across all 400k papers
        pct_1_global = (y_T / total_pos) * 100.0
        pct_0_global = (y_ST / total_neg) * 100.0

        # Step 3(c): Binomial Log-Likelihood Ratio within language
        p_T = (y_T + 1.0) / (n_T + 2.0)
        p_ST = (y_ST + 1.0) / (n_ST + 2.0)
        p_pool = (y_T + y_ST + 2.0) / (n_T + n_ST + 4.0)

        ll_T = xlogy(y_T + 1.0, p_T / p_pool) + xlogy(n_T - y_T + 1.0, (1.0 - p_T) / (1.0 - p_pool))
        ll_ST = xlogy(y_ST + 1.0, p_ST / p_pool) + xlogy(n_ST - y_ST + 1.0, (1.0 - p_ST) / (1.0 - p_pool))
        klr_score = 2.0 * (ll_T + ll_ST)

        gram_type = [
            f"{w.count(' ') + 1}-grams" if lang not in {"zh", "ja", "ko"} else f"{len(w)}-char"
            for w in vocab
        ]

        lang_df = pd.DataFrame({
            "language": lang,
            "gram_type": gram_type,
            "keyword": vocab,
            "pct_1_in_lang": np.round(pct_1_lang, 2),
            "pct_0_in_lang": np.round(pct_0_lang, 2),
            "pct_1_global": np.round(pct_1_global, 3),
            "pct_0_global": np.round(pct_0_global, 3),
            "count_1": y_T,
            "count_0": y_ST,
            "lift_in_lang": np.round((pct_1_lang + 0.05) / (pct_0_lang + 0.05), 2),
            "klr_likelihood": np.round(klr_score, 2)
        })

        # Step 4: Separate into Target (T: 1) and Non-Target (S \ T: 0) lists
        all_tables_T.append(lang_df[lang_df["pct_1_in_lang"] > lang_df["pct_0_in_lang"]])
        all_tables_ST.append(lang_df[lang_df["pct_0_in_lang"] > lang_df["pct_1_in_lang"]])

    table_T = pd.concat(all_tables_T, ignore_index=True).sort_values(
        by=["language", "klr_likelihood"], ascending=[True, False]
    )
    table_ST = pd.concat(all_tables_ST, ignore_index=True).sort_values(
        by=["language", "klr_likelihood"], ascending=[True, False]
    )

    return table_T, table_ST