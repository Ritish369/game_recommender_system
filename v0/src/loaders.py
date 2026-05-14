import pandas as pd
from .config import (
    GAMES_CSV,
    INTERACTIONS_CHUNK_SIZE,
    INTERACTIONS_CSV,
    METADATA_JSON,
    MIN_INTERACTIONS,
    PROCESSED_GAMES_CSV,
    PROCESSED_INTERACTIONS_CSV,
    RNG,
    TARGET_ACTIVE_USERS,
    TEST_SIMULATIONS_CSV,
    TRAIN_SIMULATIONS_CSV,
    USER_CHUNK_SIZE,
    USER_PROFILES_CSV,
    USERS_CSV,
)


def drop_unnamed_columns(frame: pd.DataFrame) -> pd.DataFrame:
    unnamed_cols = [col for col in frame.columns if str(col).startswith("Unnamed:")]
    if unnamed_cols:
        frame = frame.drop(columns=unnamed_cols)
    return frame


def load_games(data_csv=GAMES_CSV) -> pd.DataFrame:
    return drop_unnamed_columns(pd.read_csv(data_csv))


def load_metadata(metadata_json=METADATA_JSON) -> pd.DataFrame:
    return drop_unnamed_columns(pd.read_json(metadata_json, lines=True))


def sample_active_users(
    users_csv=USERS_CSV,
    min_interactions: int = MIN_INTERACTIONS,
    target_active_users: int = TARGET_ACTIVE_USERS,
    user_chunk_size: int = USER_CHUNK_SIZE,
    rng=RNG,
) -> set[int]:
    reservoir: list[int] = []
    eligible_users_seen = 0

    for user_chunk in pd.read_csv(
        users_csv,
        usecols=["user_id", "reviews"],
        chunksize=user_chunk_size,
    ):
        eligible_user_ids = user_chunk.loc[
            user_chunk["reviews"] >= min_interactions,
            "user_id",
        ]
        for user_id in eligible_user_ids:
            eligible_users_seen += 1
            if len(reservoir) < target_active_users:
                reservoir.append(int(user_id))
                continue

            replacement_idx = rng.randint(0, eligible_users_seen - 1)
            if replacement_idx < target_active_users:
                reservoir[replacement_idx] = int(user_id)

    return set(reservoir)


def load_sampled_interactions(
    users: set[int],
    interactions_csv=INTERACTIONS_CSV,
    interactions_chunk_size: int = INTERACTIONS_CHUNK_SIZE,
) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []

    for chunk in pd.read_csv(interactions_csv, chunksize=interactions_chunk_size):
        sampled_chunk = chunk[chunk["user_id"].isin(users)]
        if not sampled_chunk.empty:
            chunks.append(sampled_chunk)

    if not chunks:
        return pd.DataFrame()

    return drop_unnamed_columns(pd.concat(chunks, ignore_index=True))


def load_raw_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    games = load_games()
    metadata = load_metadata()
    sampled_users = sample_active_users()
    interactions = load_sampled_interactions(sampled_users)
    return games, metadata, interactions


def load_processed_games(processed_games_csv=PROCESSED_GAMES_CSV) -> pd.DataFrame:
    games = drop_unnamed_columns(pd.read_csv(processed_games_csv))
    if "date_release" in games.columns:
        games["date_release"] = pd.to_datetime(games["date_release"])
    return games


def load_processed_interactions(
    processed_interactions_csv=PROCESSED_INTERACTIONS_CSV,
) -> pd.DataFrame:
    interactions = drop_unnamed_columns(pd.read_csv(processed_interactions_csv))
    if "date" in interactions.columns:
        interactions["date"] = pd.to_datetime(interactions["date"])
    return interactions


def load_train_simulations(train_simulations_csv=TRAIN_SIMULATIONS_CSV) -> pd.DataFrame:
    return drop_unnamed_columns(pd.read_csv(train_simulations_csv))


def load_test_simulations(test_simulations_csv=TEST_SIMULATIONS_CSV) -> pd.DataFrame:
    return drop_unnamed_columns(pd.read_csv(test_simulations_csv))


def load_user_profiles(user_profiles_csv=USER_PROFILES_CSV) -> pd.DataFrame:
    return drop_unnamed_columns(pd.read_csv(user_profiles_csv))


# Added by Codex: keep the original function names as thin wrappers so the
# notebook can migrate gradually without breaking existing imports.
def games_loader(data_csv=GAMES_CSV) -> pd.DataFrame:
    return load_games(data_csv)


def metadata_loader(metadata_json=METADATA_JSON) -> pd.DataFrame:
    return load_metadata(metadata_json)


def users_loader(
    users_csv=USERS_CSV,
    min_interactions: int = MIN_INTERACTIONS,
    target_active_users: int = TARGET_ACTIVE_USERS,
    user_chunk_size: int = USER_CHUNK_SIZE,
    rng=RNG,
) -> set[int]:
    return sample_active_users(
        users_csv=users_csv,
        min_interactions=min_interactions,
        target_active_users=target_active_users,
        user_chunk_size=user_chunk_size,
        rng=rng,
    )


def interactions_loader(
    interactions_csv=INTERACTIONS_CSV,
    interactions_chunk_size: int = INTERACTIONS_CHUNK_SIZE,
    users: set[int] | None = None,
) -> pd.DataFrame:
    users = users or set()
    return load_sampled_interactions(
        users=users,
        interactions_csv=interactions_csv,
        interactions_chunk_size=interactions_chunk_size,
    )
