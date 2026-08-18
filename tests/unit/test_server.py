from fastapi.testclient import TestClient

from orbit.server.app import app


def test_server_health_and_system_endpoints():
    client = TestClient(app)

    # 1. Health check
    res_health = client.get("/health")
    assert res_health.status_code == 200
    assert res_health.json()["status"] == "healthy"

    # 2. System info
    res_sys = client.get("/api/system")
    assert res_sys.status_code == 200
    data = res_sys.json()
    assert "hardware_info" in data
    assert "git_commit" in data
    assert "os" in data["hardware_info"]


def test_server_dashboard_endpoint():
    client = TestClient(app)
    res = client.get("/dashboard")
    assert res.status_code == 200
    assert "ORBIT" in res.text
    assert "RESEARCH CONSOLE" in res.text


def test_server_train_and_experiment_lifecycle(tmp_path):
    client = TestClient(app)
    out_dir = str(tmp_path / "server_experiments")

    # Trigger training via API
    payload = {
        "steps": 2,
        "strategy": "adaptive",
        "seed": 42,
        "provider": "mock",
        "output_dir": out_dir,
    }
    res_train = client.post("/api/train", json=payload)
    assert res_train.status_code == 200
    train_data = res_train.json()
    assert "experiment_id" in train_data
    assert train_data["duration_sec"] >= 0.0

    exp_id = train_data["experiment_id"]
    assert exp_id.startswith("exp_api_adaptive_")
