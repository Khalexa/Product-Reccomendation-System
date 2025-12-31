# Cold Boot Instructions for Product Recommendation System

## Quick Start

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

**Expected output:**
- Flask, pandas, scikit-learn, joblib, requests installed
- No errors

---

### Step 2: Generate Synthetic Sample Data
```bash
python scripts/generate_synthetic_sample.py
```

**Expected output:**
```
✅ Generated 500 items
✅ Generated 200 users
✅ Generated 5000 events
✅ Sample saved to data/processed/
```

**What it creates:**
- `data/processed/sample_items.csv` (500 products with category, brand, image URL)
- `data/processed/sample_events.csv` (5000 user interactions: views, clicks, purchases)
- `data/processed/sample_users.csv` (200 users with segment: bargain/premium/tech/home/outdoor)

---

### Step 3: Train Model & Prewarm Cache
```bash
python scripts/prewarm_cache.py --top 20 --k 6
```

**Expected output:**
```
✅ Loaded sample data: 200 users, 500 items, 5000 events
✅ Trained recommender
✅ Computing recommendations for top 20 users...
✅ Persisted to SQLite cache
✅ Cache ready with 20 users
```

**What it does:**
- Trains user-user and item-item similarity matrices
- Pre-computes top-6 recommendations for top 20 users
- Saves to `cache.db` for instant retrieval
- Downloads product images to `static/thumbs/`

---

### Step 4: Start Flask Server
```bash
python app.py
```

**Expected output:**
```
 * Running on http://127.0.0.1:5000
 * Debug mode: ON
WARNING in flask.app: ...
WARNING: This is a development server. Do not use in production.
```

**Server is now ready!** Do NOT close this terminal.

---

### Step 5: Open Browser Demo
1. Open http://127.0.0.1:5000 in browser
2. You'll see:
   - **Top Users Dropdown** (left panel) — explore pre-computed recommendations
   - **Sign In Button** (top right)
   - **Item Grid** (center) — showing 10 random products
   - **Loader Status Panel** (bottom right) — shows "Sample Dataset, 5000 events"

---

## Demo Flow (What to Show Your Professor)

### Demo 1: User-Based Recommendations (Precomputed)
1. Click dropdown "Select User"
2. Choose any user (e.g., "User 2")
3. See 6 products recommended based on similar users' preferences
4. **Explain:** "This used collaborative filtering to find users like User 2, then recommended items those similar users liked."

**Key point:** Instant results (cached in database).

---

### Demo 2: Session-Based Recommendations (Real-Time)
1. Click **"Sign In"** button
2. **Session Panel** appears on right showing:
   - ✅ Signed In
   - Items: 0
   - Total Clicks: 0
   - Total Views: 0

3. **Click 2-3 product cards** in the grid
4. Watch in real-time:
   - Session panel updates (Items count increases)
   - CTR display shows per-item click-through rate (e.g., "Item 100162: 50%")
   - Item Grid refreshes with recommendations similar to what you clicked

**Key point:** No user database, no persistent storage. Just in-memory session tracking.

5. **Click same item twice**
   - CTR for that item updates to reflect additional click
   - Weights = (views + 2×clicks) × e^(-0.01×time_elapsed)
   - Recency decay favors recent interactions

6. Click **"Sign Out"** → Session clears instantly

**Explain:** "All recommendations are in-memory ephemeral sessions. When you sign out, all your data is gone."

---

### Demo 3: Loader Switch (Optional - Full Dataset)
1. Click **"Switch to Full Dataset"** button (if available)
2. Server retrains on full RetailRocket data (if `backend/data_loader.py` + data exists)
3. Event count jumps from 5K to 2.7M
4. Pre-computed recommendations change

**Key point:** System supports both sample and production datasets with a toggle.

---

### Demo 4: API Inspection (for Developers)
Open a second terminal and run:

```bash
curl http://127.0.0.1:5000/loader_status
```

**Expected response:**
```json
{
  "events_count": 5000,
  "use_full_dataset": false
}
```

This shows your professor the REST API layer.

---

## Architecture Explanation

### Why Session-Based?
Traditional e-commerce systems track user IDs in databases. That's storage overhead. Our system **proves the ML works in seconds** by using disposable sessions—no user DB, all ephemeral. Perfect for demos and proof-of-concept.

### Why Item-Item CF for Sessions?
When you click shoes, we find items similar to shoes. Fast to compute, interpretable. Better than user-based CF for quick sessions where we have limited interaction history.

### Why Recency Decay?
A click 1 minute ago matters more than a click 10 minutes ago. We use exponential decay: e^(-0.01×time). Simple but effective.

### Why Prewarmed Cache?
Top users get recommendations pre-computed in the background. When they visit, it's instant. Background daemon runs every 24h to keep cache fresh.

## Full Workflow (All Steps Combined)

```bash
# Terminal 1: Setup
pip install -r requirements.txt
python scripts/generate_synthetic_sample.py
python scripts/prewarm_cache.py --top 20 --k 6

# Terminal 1: Start server (keep running)
python app.py

# Browser (in separate tab)
# Open http://127.0.0.1:5000
# Demo: Sign In → Click Items → Watch CTR & Recommendations Update → Sign Out

# Terminal 2 (optional, for API inspection)
curl http://127.0.0.1:5000/loader_status
curl http://127.0.0.1:5000/get_recommendations/2  # See precomputed recs for user 2
```

---

