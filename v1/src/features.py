# features.py — Feature engineering pipeline for v1 item tower.
# Text cleaning → SentenceTransformer embeddings → block extraction → weighted merge.
# Produces the 1,173-dim static feature matrix fed to the item tower.

from __future__ import annotations

import string

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler, normalize


def engineer_game_features(games: pd.DataFrame) -> pd.DataFrame:
    """Drop unused price cols, one-hot rating, bool, log1p features, date parsing."""
    games = games.copy()
    games.drop(columns=["price_original", "discount"], inplace=True)

    bool_cols = ["win", "mac", "linux", "steam_deck"]
    games[bool_cols] = games[bool_cols].astype("bool")

    games = pd.get_dummies(games, columns=["rating"], dtype="bool")

    games["user_reviews"] = np.log1p(games["user_reviews"].clip(lower=0))
    games["price_final"] = np.log1p(games["price_final"].clip(lower=0))

    games["date_release"] = pd.to_datetime(games["date_release"], errors="coerce")
    games.dropna(subset=["date_release"], inplace=True)
    return games


def rescale_positive_ratio(games: pd.DataFrame) -> pd.DataFrame:
    """Store positive_ratio as 0–100 → rescale to [0, 1] for feature parity."""
    games = games.copy()
    games["positive_ratio"] = games["positive_ratio"] / 100.0
    return games


def enrich_descriptions_with_title(
    games: pd.DataFrame, games_metadata: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge game title into metadata so the SentenceTransformer sees the name.
    
    Title column is dropped from games table afterwards to avoid duplication.
    """
    games_metadata = games_metadata.merge(
        games[["app_id", "title"]],
        on="app_id",
        how="inner",
        validate="one_to_one",
    )
    games.drop(columns=["title"], inplace=True)
    return games, games_metadata


def clean_text_for_embeddings(
    games_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Strip HTML, URLs, punctuation; collapse whitespace for tags + descriptions + title.

    Tags carry broad genre signals; descriptions carry gameplay/story/text.
    """
    meta = games_metadata.copy()

    # Concatenate title into description for richer embedding context.
    meta["description_clean"] = (
        meta["description"].fillna("").astype(str)
        + " "
        + meta["title"].fillna("").astype(str)
    ).str.lower()
    meta.drop(columns=["title"], inplace=True)

    # Join tag list into a single space-separated string.
    meta["tags_clean"] = meta["tags"].apply(
        lambda tags: " ".join(str(tag) for tag in tags)
    )

    # Clean tags: remove punctuation, collapse whitespace.
    meta["tags_clean"] = meta["tags_clean"].apply(
        lambda text: text.translate(str.maketrans("", "", string.punctuation))
    )
    meta["tags_clean"] = (
        meta["tags_clean"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    # Clean descriptions: strip HTML tags.
    meta["description_clean"] = meta["description_clean"].str.replace(
        r"<[^>]+>", " ", regex=True
    )
    # Clean descriptions: remove URLs.
    meta["description_clean"] = meta["description_clean"].str.replace(
        r"http\S+|www\.\S+", " ", regex=True
    )
    # Clean descriptions: remove punctuation.
    meta["description_clean"] = meta["description_clean"].apply(
        lambda text: text.translate(str.maketrans("", "", string.punctuation))
    )
    # Collapse whitespace.
    meta["description_clean"] = (
        meta["description_clean"]
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return meta


def encode_tags(
    games_metadata: pd.DataFrame,
    model_name: str = "sentence-transformers/all-MiniLM-L12-v2",
) -> pd.DataFrame:
    """MiniLM-L12 (384-dim) encodes cleaned tag strings. Zeros out items without tags."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        games_metadata["tags_clean"].tolist(),
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    # Items with no tags get a zero vector — avoids noise from empty-string encodings.
    embeddings[games_metadata["has_tags"].to_numpy() == 0] = 0.0

    tags_df = pd.DataFrame(
        embeddings,
        columns=[f"tag_feature_{i}" for i in range(embeddings.shape[1])],
    )
    tags_df.insert(0, "app_id", games_metadata["app_id"].values)
    return tags_df


def encode_descriptions(
    games_metadata: pd.DataFrame,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
) -> pd.DataFrame:
    """MPNet (768-dim) encodes cleaned descriptions + title. Zeros out missing-description items."""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        games_metadata["description_clean"].tolist(),
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings[games_metadata["has_description"].to_numpy() == 0] = 0.0

    desc_df = pd.DataFrame(
        embeddings,
        columns=[f"description_feature_{i}" for i in range(embeddings.shape[1])],
    )
    desc_df.insert(0, "app_id", games_metadata["app_id"].values)
    return desc_df


def merge_feature_tables(
    games: pd.DataFrame,
    tags_df: pd.DataFrame,
    desc_df: pd.DataFrame,
    metadata_availability: pd.DataFrame,
) -> pd.DataFrame:
    """Inner-join games, tags, descriptions, and availability flags into one feature table."""
    return (
        games.merge(tags_df, on="app_id", how="inner", validate="one_to_one")
        .merge(desc_df, on="app_id", how="inner", validate="one_to_one")
        .merge(
            metadata_availability,
            on="app_id",
            how="inner",
            validate="one_to_one",
        )
    )


def get_feature_blocks(
    item_features_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Partition columns into four blocks for independent scaling + weighting.

    Returns (X_numeric, X_binary, X_tags, X_descriptions, game_ids, release_dates).
    """
    numeric_cols = ["positive_ratio", "user_reviews", "price_final"]
    binary_cols = [
        col
        for col in item_features_df.columns
        if col.startswith("rating_")
        or col in [
            "win", "mac", "linux", "steam_deck",
            "has_description", "has_tags", "has_text_metadata",
        ]
    ]
    tag_cols = [
        col for col in item_features_df.columns if col.startswith("tag_feature_")
    ]
    desc_cols = [
        col
        for col in item_features_df.columns
        if col.startswith("description_feature_")
    ]

    game_ids = item_features_df["app_id"].values
    release_dates = item_features_df["date_release"].values

    X_numeric = item_features_df[numeric_cols].to_numpy(dtype=np.float32)
    X_binary = item_features_df[binary_cols].to_numpy(dtype=np.float32)
    X_tags = item_features_df[tag_cols].to_numpy(dtype=np.float32)
    X_descriptions = item_features_df[desc_cols].to_numpy(dtype=np.float32)

    return X_numeric, X_binary, X_tags, X_descriptions, game_ids, release_dates


def _get_final_feature_colnames(item_features_df: pd.DataFrame) -> list[str]:
    """Reconstruct ordered column names matching the weighted concatenation."""
    numeric_cols = ["positive_ratio", "user_reviews", "price_final"]
    binary_cols = [
        col
        for col in item_features_df.columns
        if col.startswith("rating_")
        or col in [
            "win", "mac", "linux", "steam_deck",
            "has_description", "has_tags", "has_text_metadata",
        ]
    ]
    tag_cols = [
        col for col in item_features_df.columns if col.startswith("tag_feature_")
    ]
    desc_cols = [
        col
        for col in item_features_df.columns
        if col.startswith("description_feature_")
    ]
    return numeric_cols + binary_cols + tag_cols + desc_cols


def build_weighted_item_matrix(
    item_features_df: pd.DataFrame,
) -> pd.DataFrame:
    """L2-normalize each feature block independently, apply weights, concatenate.
    
    Block weights: 15% numeric, 15% binary, 25% tags, 45% descriptions.
    Final concatenation is L2-normalized for ANN-friendly inner products (= cosine sim).
    """
    from .config import (
        BINARY_WEIGHT,
        DESCRIPTION_WEIGHT,
        METADATA_WEIGHT,
        TAG_WEIGHT,
    )

    X_numeric, X_binary, X_tags, X_descriptions, game_ids, release_dates = (
        get_feature_blocks(item_features_df)
    )

    numeric_scaler = StandardScaler()
    X_numeric = normalize(numeric_scaler.fit_transform(X_numeric))
    X_binary = normalize(X_binary) if X_binary.shape[1] else X_binary

    X_items = np.concatenate(
        [
            METADATA_WEIGHT * X_numeric,
            BINARY_WEIGHT * X_binary,
            TAG_WEIGHT * X_tags,
            DESCRIPTION_WEIGHT * X_descriptions,
        ],
        axis=1,
    )
    X_items = normalize(X_items)  # final L2-normalize for dot-product equivalence

    final_cols = _get_final_feature_colnames(item_features_df)
    result = pd.DataFrame(X_items, columns=final_cols)
    result.insert(0, "app_id", game_ids)
    result.insert(1, "date_release", release_dates)
    result["date_release"] = pd.to_datetime(result["date_release"])
    return result
