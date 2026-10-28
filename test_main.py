from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "Food Delivery Time Prediction API is running"
    }


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_endpoint():
    payload = {
        "delivery_person_age": 28,
        "delivery_person_ratings": 4.7,
        "restaurant_latitude": 12.9716,
        "restaurant_longitude": 77.5946,
        "delivery_location_latitude": 12.9352,
        "delivery_location_longitude": 77.6245,
        "type_of_order": 1,
        "type_of_vehicle": 0
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    assert "predicted_delivery_time_minutes" in result
    assert isinstance(
        result["predicted_delivery_time_minutes"], (int, float)
    )


def test_predict_rejects_invalid_age():
    payload = {
        "delivery_person_age": 150,
        "delivery_person_ratings": 4.7,
        "restaurant_latitude": 12.9716,
        "restaurant_longitude": 77.5946,
        "delivery_location_latitude": 12.9352,
        "delivery_location_longitude": 77.6245,
        "type_of_order": 1,
        "type_of_vehicle": 0
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
