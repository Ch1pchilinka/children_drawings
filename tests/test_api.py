from children_drawings.api import _health_sync, app


def test_api_routes_are_registered():
    paths = {route.path for route in app.routes}

    assert "/" in paths
    assert "/health" in paths
    assert "/predict" in paths
    assert "/predict/batch" in paths
    assert "/predict/base64" in paths


def test_api_health_smoke():
    response = _health_sync()

    assert "status" in response
    assert response["model_name"] == "children_drawings"
    assert "server_live" in response
