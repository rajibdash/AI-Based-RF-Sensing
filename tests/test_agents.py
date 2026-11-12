import pytest

from rf_sensing.agents import SpectrumPolicyAgent


def test_policy_agent_is_conservative_and_advisory():
    policy = SpectrumPolicyAgent()
    assert policy.recommend(0.95) == "protect"
    assert policy.recommend(0.50) == "resense"
    assert policy.recommend(0.05) == "access_candidate"


def test_policy_agent_rejects_invalid_probability():
    with pytest.raises(ValueError):
        SpectrumPolicyAgent().recommend(1.1)
