from pathlib import Path
import configparser
import yaml
import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
PYTEST_INI_PATH = REPO_ROOT / "pytest.ini"
CI_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def test_pytest_ini_exists_and_parses():
    assert PYTEST_INI_PATH.is_file(), "pytest.ini must exist at project root"
    config = configparser.ConfigParser()
    config.read(PYTEST_INI_PATH)
    assert "pytest" in config, "pytest.ini must define a [pytest] section"
    assert config["pytest"]["testpaths"] == "tests"
    assert config["pytest"]["pythonpath"] == "."


def test_pytest_ini_registers_slow_marker(pytestconfig):
    markers = pytestconfig.getini("markers")
    slow_markers = [m for m in markers if m.startswith("slow")]
    assert len(slow_markers) > 0, "slow marker must be registered in pytest markers"
    assert "marks tests as slow" in slow_markers[0]


def test_ci_workflow_exists():
    assert CI_WORKFLOW_PATH.is_file(), ".github/workflows/ci.yml must exist"


def test_ci_workflow_valid_yaml_syntax():
    content = CI_WORKFLOW_PATH.read_text(encoding="utf-8")
    data = yaml.safe_load(content)
    assert isinstance(data, dict), "CI workflow YAML root must parse to a dictionary"
    assert data.get("name") == "CI Pipeline"


def test_ci_workflow_triggers():
    content = CI_WORKFLOW_PATH.read_text(encoding="utf-8")
    data = yaml.safe_load(content)
    
    triggers = data.get("on") or data.get(True)  # PyYAML might parse unquoted "on" as True
    assert triggers is not None, "Workflow must define triggers for push and pull_request"
    
    # Check push branches
    assert "push" in triggers
    push_branches = triggers["push"].get("branches", [])
    assert "main" in push_branches
    assert "feature/*" in push_branches

    # Check pull_request branches
    assert "pull_request" in triggers
    pr_branches = triggers["pull_request"].get("branches", [])
    assert "main" in pr_branches
    assert "feature/*" in pr_branches


def test_ci_workflow_job_structure_and_steps():
    content = CI_WORKFLOW_PATH.read_text(encoding="utf-8")
    data = yaml.safe_load(content)
    
    jobs = data.get("jobs", {})
    assert "test" in jobs, "CI workflow must have a 'test' job"
    
    test_job = jobs["test"]
    assert test_job.get("runs-on") == "ubuntu-latest"
    assert test_job.get("env", {}).get("PYTHONPATH") == "."

    steps = test_job.get("steps", [])
    assert len(steps) >= 5, "CI workflow should define all required build/test steps"
    
    step_runs = [s.get("run", "") for s in steps if "run" in s]
    step_uses = [s.get("uses", "") for s in steps if "uses" in s]

    # Checkout
    assert any("checkout" in u for u in step_uses), "Must contain actions/checkout step"
    # Setup python
    assert any("setup-python" in u for u in step_uses), "Must contain actions/setup-python step"
    
    # Dependencies install
    assert any("pip install -r requirements.txt" in r for r in step_runs), "Must install dependencies from requirements.txt"

    # Syntax compilation
    compile_steps = [r for r in step_runs if "compileall" in r]
    assert len(compile_steps) > 0, "Must run python -m compileall syntax check"
    compile_cmd = compile_steps[0]
    for target_dir in ["api", "src", "tests", "scripts"]:
        assert target_dir in compile_cmd, f"compileall must cover {target_dir}"

    # Fast test execution
    assert any('pytest -q -m "not slow"' in r for r in step_runs), 'Must run fast test command: pytest -q -m "not slow"'

    # Full test execution
    assert any("pytest -q" in r for r in step_runs), "Must run full test command: pytest -q"


def test_ci_workflow_strictly_omits_legacy_streamlit_and_app_dir():
    content = CI_WORKFLOW_PATH.read_text(encoding="utf-8").lower()
    # Check that legacy references do not exist
    assert "streamlit" not in content, "CI workflow must not reference streamlit (ADR-0007)"
    
    # Check that 'app/' directory (Streamlit directory) is not in compileall or any step
    lines = content.splitlines()
    for line in lines:
        if "compileall" in line:
            tokens = line.split()
            assert "app" not in tokens, "compileall must not include legacy 'app' directory"


def test_invalid_yaml_syntax_fails_parser():
    malformed_yaml = """
name: Bad Workflow
on: [push
  invalid_indent: true
"""
    with pytest.raises(yaml.YAMLError):
        yaml.safe_load(malformed_yaml)


def test_invalid_ini_marker_structure_validation():
    malformed_ini = """
[pytest
testpaths = tests
"""
    config = configparser.ConfigParser()
    with pytest.raises(configparser.Error):
        config.read_string(malformed_ini)
