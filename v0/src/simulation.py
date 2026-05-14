import pandas as pd
import numpy as np
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm

from .config import MIN_INTERACTIONS, TRAIN_RATIO
from .loaders import drop_unnamed_columns


def train_test_users(
    games,
    processed_interactions,
    min_interactions: int = MIN_INTERACTIONS,
    train_ratio: float = TRAIN_RATIO,
):
    games = drop_unnamed_columns(games.copy())
    processed_interactions = drop_unnamed_columns(processed_interactions.copy())
    processed_interactions["date"] = pd.to_datetime(processed_interactions["date"])
    numeric_cols = processed_interactions.select_dtypes(include=["number"]).columns
    processed_interactions[numeric_cols] = processed_interactions[numeric_cols].apply(
        pd.to_numeric,
        downcast="float",
    )

    user_group_sizes = (
        processed_interactions.groupby("user_id", sort=False)
        .size()
        .reset_index(name="size")
        .sort_values(by="size", ascending=False)
    )
    active_user_ids = user_group_sizes[user_group_sizes["size"] >= min_interactions]["user_id"]

    user_groups_simulated_df = processed_interactions[
        processed_interactions["user_id"].isin(active_user_ids)
    ].copy()

    user_groups_simulated_df.sort_values(by=["user_id", "date"], inplace=True)
    user_groups_simulated_df.reset_index(drop=True, inplace=True)

    train = []
    test = []

    for _, group in tqdm(
        user_groups_simulated_df.groupby("user_id", sort=False),
        desc="User groups split...",
    ):
        split_idx = int(len(group) * train_ratio)
        split_idx = min(max(split_idx, 1), len(group) - 1)
        train.append(group.iloc[:split_idx])
        test.append(group.iloc[split_idx:])

    train_simu_users = pd.concat(train, ignore_index=True)
    test_simu_users = pd.concat(test, ignore_index=True)

    train_simu_users = train_simu_users[train_simu_users["app_id"].isin(games["app_id"])].copy()
    test_simu_users = test_simu_users[test_simu_users["app_id"].isin(games["app_id"])].copy()

    train_simu_users.drop(columns=["date"], inplace=True, errors="ignore")
    test_simu_users.drop(columns=["date"], inplace=True, errors="ignore")

    return train_simu_users, test_simu_users

def user_profiles(games, train_simu):
    games = drop_unnamed_columns(games.copy())
    train_simu = drop_unnamed_columns(train_simu.copy())
    train_game_vectors = games[games["app_id"].isin(train_simu["app_id"])].reset_index(drop=True)
    
    embedding_cols = [col for col in train_game_vectors.columns if col not in ["app_id", "date_release"]]

    train_merged = train_simu.merge(train_game_vectors, on="app_id", how="left")

    profiles = {}

    for userid, group in tqdm(train_merged.groupby("user_id"), desc="Users processed..."):
        vectors = group[embedding_cols].values
        weights = group["implicit_rating"].values.reshape(-1, 1)
        
        weighted_sum = (vectors * weights).sum(axis=0)
        weight_total = weights.sum()

        # Added by Codex: when a user's train split has only negative feedback,
        # fall back to the mean of seen items so the profile still exists.
        if weight_total > 0:
            user_vector = weighted_sum / weight_total
        else:
            user_vector = vectors.mean(axis=0)

        user_vector = normalize(user_vector.reshape(1, -1))[0]
        profiles[userid] = user_vector

    user_ids = list(profiles.keys())
    vectors = np.vstack(list(profiles.values()))

    user_profiles_df = pd.DataFrame(vectors, columns=embedding_cols)
    user_profiles_df.insert(0, "user_id", user_ids)

    user_profiles_df_num_cols = user_profiles_df.select_dtypes(include=["number"]).columns
    user_profiles_df[user_profiles_df_num_cols] = user_profiles_df[user_profiles_df_num_cols].apply(
        pd.to_numeric,
        downcast="float",
    )

    return user_profiles_df
