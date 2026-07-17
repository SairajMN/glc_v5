def test_gateway_does_not_own_session_13_runtime_routes(app_client):
    assert app_client.post("/v1/agent/runs", json={}).status_code == 404
    assert app_client.get("/.well-known/agent-card.json").status_code == 404
    assert app_client.get("/v1/agent/memory/search").status_code == 404
