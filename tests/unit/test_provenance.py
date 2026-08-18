from orbit.data.trajectory import Provenance
from orbit.utils.provenance import (
    capture_provenance,
    get_git_info,
    get_hardware_info,
)


def test_get_git_info():
    commit, is_dirty = get_git_info()
    assert isinstance(commit, str)
    assert isinstance(is_dirty, bool)


def test_get_hardware_info():
    info = get_hardware_info()
    assert "os" in info
    assert "python_version" in info
    assert "torch_version" in info
    assert "cuda_available" in info


def test_capture_provenance():
    prov = capture_provenance(
        model_version="test-model",
        env_version="0.1.0",
        seed=1234,
    )
    assert isinstance(prov, Provenance)
    assert prov.model_version == "test-model"
    assert prov.env_version == "0.1.0"
    assert prov.seed == 1234
    assert prov.hardware_info["python_version"] is not None

    prov_dict = prov.to_dict()
    assert prov_dict["model_version"] == "test-model"
    reconstructed = Provenance.from_dict(prov_dict)
    assert reconstructed == prov
