from children_drawings.api import app, health


def test_api_routes_are_registered():
    paths = {route.path for route in app.routes}

    assert "/health" in paths
    assert "/predict" in paths
    assert "/predict/base64" in paths


def test_api_health_smoke():
    response = health()

    assert response["status"] == "ok"
    assert "model_exists" in response
