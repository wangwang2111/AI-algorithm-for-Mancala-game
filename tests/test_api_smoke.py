# tests/test_api_smoke.py
def test_health(client):
    res = client.get("/api/health")
    assert res.status_code == 200
    data = res.get_json()
    assert data.get("status") == "ok"

def test_newgame_and_move(client):
    r = client.post("/api/newgame")
    assert r.status_code == 200
    s = r.get_json()["state"]

    # call AI move with a known agent
    r2 = client.post("/api/move", json={"state": s, "agent": "advanced"})
    assert r2.status_code == 200
    payload = r2.get_json()
    assert "action" in payload and "next_state" in payload
