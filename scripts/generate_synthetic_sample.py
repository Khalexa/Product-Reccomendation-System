# Generate synthetic sample data for testing
# Creates product items and user events with realistic IDs
import csv
import random
import pathlib
from datetime import datetime, timedelta

RAW = pathlib.Path("data/raw")
RAW.mkdir(parents=True, exist_ok=True)

def generate_items(n_items=500):
    path = RAW / "item_properties_part1.csv"
    categories = [f"cat_{i}" for i in range(1, 21)]
    segments = ['bargain', 'premium', 'tech', 'home', 'outdoor']
    seg_cat = {s: random.sample(categories, k=5) for s in segments}
    with path.open("w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["itemid", "property", "value"])  # keep similar structure
        for i in range(1, n_items+1):
            itemid = 100000 + i
            prop = "category"
            val = random.choice(categories)
            writer.writerow([itemid, prop, val])
    # Write brand properties
    path2 = RAW / "item_properties_part2.csv"
    with path2.open("w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["itemid", "property", "value"])
        for i in range(1, n_items+1):
            itemid = 100000 + i
            prop = "brand"
            val = f"brand_{random.randint(1,50)}"
            writer.writerow([itemid, prop, val])
    print(f"Wrote {n_items} items to {path} and {path2}")


def generate_events(n_users=200, n_events=5000, start_ts=None):
    path = RAW / "events.csv"
    if start_ts is None:
        start = datetime.utcnow() - timedelta(days=30)
    else:
        start = start_ts
    # Setup user segments
    segments = ['bargain', 'premium', 'tech', 'home', 'outdoor']
    users = []
    user_segment = {}
    for i in range(n_users):
        uid = 1000 + i
        seg = random.choices(segments, weights=[0.3,0.15,0.2,0.2,0.15])[0]
        users.append(uid)
        user_segment[uid] = seg

    items = [100000 + i for i in range(1, 501)]
    event_types = ["view", "addtocart", "transaction"]

    with path.open("w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp","visitorid","itemid","event","transactionid"]) 
        # Create temporal bursts: more activity during recent days for some users
        for i in range(n_events):
            # More recent activity
            if random.random() < 0.6:
                offset_seconds = random.randint(0, 7*24*3600)
            else:
                offset_seconds = random.randint(7*24*3600, 30*24*3600)
            ts = int((start + timedelta(seconds=offset_seconds)).timestamp() * 1000)
            user = random.choice(users)
            # Select items based on user segment
            seg = user_segment[user]
            if seg == 'tech':
                item = random.choice(items[300:])
            elif seg == 'premium':
                item = random.choice(items[200:])
            elif seg == 'bargain':
                item = random.choice(items[:200])
            else:
                item = random.choice(items)
            evt = random.choices(event_types, weights=[0.85, 0.10, 0.05])[0]
            tx = "" if evt != "transaction" else f"tx_{random.randint(1, max(1, n_events//50))}"
            writer.writerow([ts, user, item, evt, tx])
    print(f"Wrote {n_events} events to {path}")


if __name__ == '__main__':
    generate_items(n_items=500)
    generate_events(n_users=200, n_events=5000)
    print('Synthetic sample generation complete.')
