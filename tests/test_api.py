import os
os.environ["DDOS_API_KEY"] = "test-key"
os.environ["DDOS_DB"] = ":memory:"  # replaced below by tmp file in fixture
import pytest
from fastapi.testclient import TestClient

@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["DDOS_DB"] = str(tmp_path_factory.mktemp("db") / "audit.db")
    from backend import main
    return TestClient(main.app), main

H = {"X-API-Key": "test-key"}

def test_auth_required(client):
    c, _ = client
    assert c.get("/health").status_code == 200
    assert c.get("/api/model").status_code == 401
    assert c.get("/api/model", headers=H).status_code == 200

def test_predict_presets(client):
    c, _ = client
    presets = c.get("/api/presets", headers=H).json()
    got = {}
    for name, p in presets.items():
        r = c.post("/api/predict", headers=H, json={"features": p["values"]})
        assert r.status_code == 200, r.text
        got[name] = r.json()
    assert got["Typical HTTPS browsing"]["label"] == "BENIGN"
    assert got["HTTP flood on port 80"]["label"] == "DDoS"
    assert got["HTTP flood on port 80"]["factors"]

def test_missing_features_rejected(client):
    c, _ = client
    assert c.post("/api/predict", headers=H, json={"features": {"Destination Port": 80}}).status_code == 422

def test_batch_and_alerts(client):
    c, _ = client
    r = c.post("/api/predict/batch", headers=H, files={"file": ("s.csv", open("results/sample_flows.csv", "rb"), "text/csv")})
    assert r.status_code == 200, r.text
    b = r.json()
    assert b["rows"] == b["ddos"] + b["benign"] and b["vs_labels"]["accuracy"] > 0.95
    assert len(c.get("/api/alerts?severity=high", headers=H).json()) > 0
    assert c.get("/api/alerts.csv", headers=H).text.startswith("id,time_utc")
