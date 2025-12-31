from flask import Flask, render_template, request, jsonify
from sample_data_loader import load_events, load_items
from sample_recommender import RecommenderSystem

app = Flask(__name__)

# Global data for the demo
events_df = load_events()
model = RecommenderSystem()
model.train(events_df)

@app.route('/')
def index():
    users = events_df['user_id'].unique()[:10].tolist()
    return render_template('index.html', users=users)

@app.route('/get_recommendations/<int:user_id>')
def get_rec(user_id):
    recs = model.recommend(user_id)
    items_info = load_items(recs)
    
    result = []
    for r in recs:
        # Get a property name (like 'categoryid') for the UI name
        details = items_info[items_info['itemid'] == r]
        name = f"Item {r}" if details.empty else f"Category {details.iloc[0]['value']}"
        result.append({"id": r, "display_name": name})
        
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)