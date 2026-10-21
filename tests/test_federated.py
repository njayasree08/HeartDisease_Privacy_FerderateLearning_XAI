import numpy as np, pytest
from src.data.preprocessing import clean_frame, split_data
from src.federated.client import HospitalClient
from src.federated.server import FederatedServer
from src.federated.simulation import run_federated
from src.federated.strategy import fedavg, flower_fedavg


def test_fedavg_weighted_values():
    a = [np.array([[1., 1.]]), np.array([0.])]; b = [np.array([[3., 5.]]), np.array([4.])]
    out = fedavg([a, b], [1, 3])
    assert np.allclose(out[0], [[2.5, 4.]]) and np.allclose(out[1], [3.])


def test_fedavg_matches_flower_and_rejects_incompatible():
    ws = [[np.random.rand(3, 2), np.random.rand(2)] for _ in range(3)]; ns = [5, 20, 40]
    assert all(np.allclose(x, y) for x, y in zip(fedavg(ws, ns), flower_fedavg(ws, ns)))
    with pytest.raises(ValueError):
        fedavg([[np.zeros((2, 2))], [np.zeros((3, 2))]], [1, 1])
    with pytest.raises(ValueError):
        fedavg([], [])


def test_client_data_isolation():
    X, y = np.random.rand(20, 5).astype("float32"), np.random.randint(0, 2, 20).astype("float32")
    c = HospitalClient(0, X, y, 0.01, 1)
    public = [a for a in dir(c) if not a.startswith("_")]
    assert not any(isinstance(getattr(c, a), np.ndarray) for a in public)
    w, n, m = c.fit(c.model.get_weights(), 1, 8)
    assert n == 20 and all(isinstance(t, np.ndarray) for t in w)
    s = FederatedServer(w); s.aggregate([(w, n)])
    import inspect; assert list(inspect.signature(FederatedServer.aggregate).parameters) == ["self", "updates"]


def test_federated_run_executes_and_reports(frame):
    sp = split_data(clean_frame(frame))
    r = run_federated(sp, n_clients=3, kind="non_iid", rounds=2, local_epochs=1)
    assert len(r["history"]) == 2 and r["comm"]["raw_records_sent_to_server"] == 0
    assert sum(r["client_sizes"]) == len(sp["train"][1]) and 0 <= r["final_federated"]["roc_auc"] <= 1
    assert r["history"][-1]["status"] == "completed"
