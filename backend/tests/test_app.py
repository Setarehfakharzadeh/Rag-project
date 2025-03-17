import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index_route(client):
    """Test that the index route returns 200"""
    rv = client.get('/')
    assert rv.status_code == 200

def test_chat_route(client):
    """Test that the chat route works correctly"""
    response = client.post('/api/chat', json={
        'message': 'Hello'
    })
    assert response.status_code == 200
    json_data = response.get_json()
    assert json_data['success'] == True
    assert 'response' in json_data

def test_chat_route_no_message(client):
    """Test that the chat route handles missing message"""
    response = client.post('/api/chat', json={})
    assert response.status_code == 400 