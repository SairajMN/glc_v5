def test_dashboard_and_help_are_served(app_client):
    dashboard = app_client.get("/")
    assert dashboard.status_code == 200
    assert "GLC v5" in dashboard.text
    assert "/channels" in dashboard.text
    assert "/v1/status" in dashboard.text

    help_page = app_client.get("/help")
    assert help_page.status_code == 200
    assert "GLC v3" in help_page.text

    static = app_client.get("/static/dashboard.html")
    assert static.status_code == 200
