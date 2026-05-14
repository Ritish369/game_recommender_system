import string
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, normalize

from .config import (
    BINARY_WT,
    DESCRIPTION_MAX_DF,
    DESCRIPTION_MAX_FEATURES,
    DESCRIPTION_MIN_DF,
    DESCRIPTION_WT,
    MAX_SVD_COMPONENTS,
    METADATA_WT,
    TAG_MAX_DF,
    TAG_MIN_DF,
    TAG_WEIGHT,
)


def _reduce_text_block(matrix, prefix, app_ids):
    if matrix.shape[1] <= 1:
        dense = matrix.toarray()
        frame = pd.DataFrame(dense, columns=[f"{prefix}_feature_0"])
    else:
        n_components = min(MAX_SVD_COMPONENTS, matrix.shape[1] - 1)
        reducer = TruncatedSVD(n_components=n_components, random_state=42)
        reduced = reducer.fit_transform(matrix)
        frame = pd.DataFrame(
            reduced,
            columns=[f"{prefix}_feature_{i}" for i in range(n_components)],
        )

    frame.insert(0, "app_id", app_ids.values)
    return frame.apply(pd.to_numeric, downcast="float")

def extract_features(processed_metadata):
    processed_metadata = processed_metadata.copy()

    processed_metadata["tags"] = processed_metadata["tags"].fillna("").astype(str)
    processed_metadata["tags"] = processed_metadata["tags"].str.replace("[", "", regex=False)
    processed_metadata["tags"] = processed_metadata["tags"].str.replace("]", "", regex=False)
    processed_metadata["tags"] = processed_metadata["tags"].str.replace("'", "", regex=False)
    processed_metadata["tags"] = processed_metadata["tags"].str.strip()

    processed_metadata["description"] = (
        processed_metadata["description"].fillna("").astype(str).str.lower()
    )

    processed_metadata["tags_clean"] = processed_metadata["tags"].apply(
        lambda text: text.translate(str.maketrans("", "", string.punctuation))
    )
    processed_metadata["tags_clean"] = processed_metadata["tags_clean"].str.replace(
        r"\s+",
        " ",
        regex=True,
    ).str.strip()

    processed_metadata["description_clean"] = processed_metadata["description"].str.replace(
        r"<[^>]+>",
        " ",
        regex=True,
    )
    processed_metadata["description_clean"] = processed_metadata["description_clean"].str.replace(
        r"http\S+|www\.\S+",
        " ",
        regex=True,
    )
    processed_metadata["description_clean"] = processed_metadata["description_clean"].apply(
        lambda text: text.translate(str.maketrans("", "", string.punctuation))
    )
    processed_metadata["description_clean"] = processed_metadata["description_clean"].str.replace(
        r"\s+",
        " ",
        regex=True,
    ).str.strip()

    tag_vectorizer = TfidfVectorizer(min_df=TAG_MIN_DF, max_df=TAG_MAX_DF)
    tag_feature_vectors = tag_vectorizer.fit_transform(processed_metadata["tags_clean"])

    description_vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=DESCRIPTION_MIN_DF,
        max_df=DESCRIPTION_MAX_DF,
        max_features=DESCRIPTION_MAX_FEATURES,
        ngram_range=(1, 2),
    )
    description_feature_vectors = description_vectorizer.fit_transform(
        processed_metadata["description_clean"]
    )

    tags_reduced = _reduce_text_block(tag_feature_vectors, "tag", processed_metadata["app_id"])
    descriptions_reduced = _reduce_text_block(
        description_feature_vectors,
        "description",
        processed_metadata["app_id"],
    )

    return tags_reduced, descriptions_reduced

def build_item_vectors(processed_games, tags_vectors, descriptions_vectors):
    final_games_transformed = (
        processed_games.merge(tags_vectors, on="app_id", how="inner")
        .merge(descriptions_vectors, on="app_id", how="inner")
        .copy()
    )

    numeric_cols = ["positive_ratio", "user_reviews", "price_final"]
    binary_cols = [
        col
        for col in final_games_transformed.columns
        if col.startswith("rating_") or col in ["win", "mac", "linux", "steam_deck"]
    ]
    tag_cols = [col for col in final_games_transformed.columns if col.startswith("tag_feature_")]
    description_cols = [
        col
        for col in final_games_transformed.columns
        if col.startswith("description_feature_")
    ]

    game_ids = final_games_transformed["app_id"].values
    dates = final_games_transformed["date_release"].values

    X_num = final_games_transformed[numeric_cols].to_numpy()
    X_binary = final_games_transformed[binary_cols].to_numpy()
    X_tags = final_games_transformed[tag_cols].to_numpy()
    X_description = final_games_transformed[description_cols].to_numpy()

    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_num)

    # Added by Codex: normalize each dense block before weighting so cosine
    # similarity reflects signal balance instead of raw block magnitude.
    X_num_block = normalize(X_num_scaled)
    X_tag_block = normalize(X_tags)
    X_description_block = normalize(X_description)

    X_final = np.concatenate([
        METADATA_WT * X_num_block,
        BINARY_WT * X_binary,
        TAG_WEIGHT * X_tag_block,
        DESCRIPTION_WT * X_description_block,
    ], axis=1)

    X_final = normalize(X_final)

    final_feature_cols = numeric_cols + binary_cols + tag_cols + description_cols
    final_games_transformed = pd.DataFrame(X_final, columns=final_feature_cols)
    final_games_transformed.insert(0, "app_id", game_ids)
    final_games_transformed.insert(1, "date_release", dates)
    final_games_transformed["date_release"] = pd.to_datetime(final_games_transformed["date_release"])

    numeric_cols = final_games_transformed.select_dtypes(include=["number"]).columns
    final_games_transformed[numeric_cols] = final_games_transformed[numeric_cols].apply(
        pd.to_numeric,
        downcast="float",
    )

    return final_games_transformed
