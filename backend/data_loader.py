import pandas as pd
import os

RAW_PATH = "data/raw/"

def load_events():
    events_path = os.path.join(RAW_PATH, "events.csv")
    df = pd.read_csv(events_path)
    # Rename columns to standard names
    df = df.rename(columns={
        "visitorid": "user_id",
        "itemid": "product_id",
        "event": "interaction_type"
    })
    # Optional: map event_type to weights
    event_weights = {"view": 1, "addtocart": 3, "transaction": 5}
    df["weight"] = df["interaction_type"].map(event_weights)
    return df

def load_items():
    items_path = os.path.join(RAW_PATH, "item_properties_part1.csv")
    df1 = pd.read_csv(items_path)
    items_path2 = os.path.join(RAW_PATH, "item_properties_part2.csv")
    df2 = pd.read_csv(items_path2)
    return pd.concat([df1, df2], ignore_index=True)

def load_categories():
    categories_path = os.path.join(RAW_PATH, "category_tree.csv")
    return pd.read_csv(categories_path)
