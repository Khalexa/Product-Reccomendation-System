import pytest
from app import app, MODEL_PATH
from sample_data_loader import load_events

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client


def test_index(client):
    res = client.get('/')
    assert res.status_code == 200


def test_get_recs_and_cache_endpoints(client):
    # Load some events to ensure model has users
    events = load_events(sample_frac=0.2, max_users=10, max_items=20, nrows=2000)
    if events.empty:
        pytest.skip('no events')
    user = int(events['user_id'].iloc[0])
    res = client.get(f'/get_recommendations/{user}')
    assert res.status_code == 200
    data = res.get_json()
    assert isinstance(data, list)
    # Test cache status
    res2 = client.get(f'/cache_status/{user}')
    assert res2.status_code == 200
    js = res2.get_json()
    assert 'cached' in js
    # Refresh recs
    res3 = client.post(f'/refresh_recs/{user}')
    assert res3.status_code == 200
    js3 = res3.get_json()
    assert 'recs' in js3
*** End Patch