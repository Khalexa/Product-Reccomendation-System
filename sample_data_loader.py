import pandas as pd
import os

RAW_PATH = "data/raw/"

def load_events(sample_frac=0.01, max_users=100, max_items=100, nrows=50000):
    """Load sample of events with configurable limits."""
    events_path = os.path.join(RAW_PATH, "events.csv")

    df = pd.read_csv(events_path, nrows=nrows)

    df = df.rename(columns={
        "visitorid": "user_id",
        "itemid": "product_id",
        "event": "interaction_type"
    })

    # Weight interactions by type
    event_weights = {"view": 1, "addtocart": 3, "transaction": 5}
    df["weight"] = df["interaction_type"].map(event_weights)

    # Filter to top users and items
    top_users = df["user_id"].value_counts().nlargest(max_users).index
    top_items = df["product_id"].value_counts().nlargest(max_items).index
    df = df[df["user_id"].isin(top_users) & df["product_id"].isin(top_items)]

    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
        
    return df

def load_items(relevant_product_ids=None, nrows=100000):
    """Load item properties and add readable display names for the demo UI.

    - `relevant_product_ids`: optional list of item ids to filter to recommended items
    - `nrows`: how many rows to read from the (first) properties file
    """
    path1 = os.path.join(RAW_PATH, "item_properties_part1.csv")
    path2 = os.path.join(RAW_PATH, "item_properties_part2.csv")

    # Read available property files (bounded) and concat
    try:
        df1 = pd.read_csv(path1, nrows=nrows)
    except Exception:
        df1 = pd.DataFrame()

    try:
        df2 = pd.read_csv(path2, nrows=nrows)
    except Exception:
        df2 = pd.DataFrame()

    if not df1.empty and not df2.empty:
        df = pd.concat([df1, df2], ignore_index=True)
    elif not df1.empty:
        df = df1
    else:
        df = df2

    # Ensure item id column exists and is named `itemid`
    if 'itemid' not in df.columns and 'id' in df.columns:
        df = df.rename(columns={'id': 'itemid'})

    # Create a human-friendly display name for the UI
    if 'value' in df.columns:
        df['display_name'] = df['value'].astype(str)
    else:
        df['display_name'] = df['itemid'].apply(lambda x: f"Item {x}")

    # If relevant ids provided, filter to them (preserve order if possible)
    if relevant_product_ids is not None:
        df = df[df['itemid'].isin(relevant_product_ids)].reset_index(drop=True)

    return df

def load_categories():
    categories_path = os.path.join(RAW_PATH, "category_tree.csv")
    return pd.read_csv(categories_path)