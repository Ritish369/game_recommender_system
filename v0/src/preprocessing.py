import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from .config import FUNNY_WT, HELPFUL_WT, HOURS_WT


def process_games_and_metadata(games, games_metadata):
    games = games.copy()
    games_metadata = games_metadata.copy()

    tags = games_metadata["tags"].fillna("").astype(str)
    descriptions = games_metadata["description"].fillna("").astype(str).str.strip()
    missing_text_mask = (tags.str.len() == 0) & (descriptions == "")
    missing_text_app_ids = set(games_metadata.loc[missing_text_mask, "app_id"])

    games = games[~games["app_id"].isin(missing_text_app_ids)].copy()
    games_metadata = games_metadata[~games_metadata["app_id"].isin(missing_text_app_ids)].copy()

    games.drop_duplicates(["title", "date_release"], inplace=True)
    games.drop(columns=["title", "price_original", "discount"], inplace=True, errors="ignore")

    games["user_reviews"] = np.log1p(games["user_reviews"])
    games["price_final"] = np.log1p(games["price_final"])
    games["date_release"] = pd.to_datetime(games["date_release"])

    bool_cols = ["win", "mac", "linux", "steam_deck"]
    games[bool_cols] = games[bool_cols].astype(int)

    games = pd.get_dummies(games, columns=["rating"], dtype=int)

    numeric_cols = games.select_dtypes(include=["number"]).columns
    games[numeric_cols] = games[numeric_cols].apply(pd.to_numeric, downcast="float")

    games.reset_index(drop=True, inplace=True)
    games_metadata.reset_index(drop=True, inplace=True)

    return games, games_metadata

def process_interactions(interactions):
    interactions = interactions.copy()
    interactions.drop(columns=["review_id"], inplace=True, errors="ignore")

    interactions["hours_transformed"] = np.log1p(interactions["hours"])
    interactions["helpful_transformed"] = np.log1p(interactions["helpful"])
    interactions["funny_transformed"] = np.log1p(interactions["funny"])
    interactions.drop(columns=["helpful", "funny", "hours"], inplace=True, errors="ignore")

    helpful_upper = interactions["helpful_transformed"].quantile(0.99)
    interactions["helpful_clipped"] = interactions["helpful_transformed"].clip(upper=helpful_upper)

    funny_upper = interactions["funny_transformed"].quantile(0.99)
    interactions["funny_clipped"] = interactions["funny_transformed"].clip(upper=funny_upper)

    interactions.drop(columns=["helpful_transformed", "funny_transformed"], inplace=True)

    scaler = MinMaxScaler()
    scaled_cols = ["hours_transformed", "helpful_clipped", "funny_clipped"]
    interactions[scaled_cols] = scaler.fit_transform(interactions[scaled_cols])

    interactions["implicit_rating"] = (
        HOURS_WT * interactions["hours_transformed"]
        + HELPFUL_WT * interactions["helpful_clipped"]
        + FUNNY_WT * interactions["funny_clipped"]
    )

    # Added by Codex: negative reviews still matter as seen items, but they
    # should not push a user profile toward a game the user did not recommend.
    interactions.loc[interactions["is_recommended"] == 0, "implicit_rating"] = 0

    interactions.drop(columns=scaled_cols, inplace=True)

    interactions["date"] = pd.to_datetime(interactions["date"])

    numeric_cols = interactions.select_dtypes(include=["number"]).columns
    interactions[numeric_cols] = interactions[numeric_cols].apply(pd.to_numeric, downcast="float")

    interactions.reset_index(drop=True, inplace=True)

    return interactions
