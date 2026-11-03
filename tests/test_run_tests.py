import re
from pathlib import Path

import pytest

import run_tests

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "target" / "scratch-tests" / "run_tests"


def test_zero_exit_is_not_a_crash():
    assert run_tests.is_crash(0) is False


def test_plain_nonzero_exit_is_not_a_crash():
    assert run_tests.is_crash(65) is False


def test_signal_death_is_a_crash():
    assert run_tests.is_crash(-11) is True


def test_windows_access_violation_is_a_crash():
    assert run_tests.is_crash(0xC0000005) is True


def test_missing_exit_status_is_a_crash():
    assert run_tests.is_crash(None) is True


def test_access_violation_is_described():
    assert "access violation" in run_tests.describe_exit(0xC0000005)


def test_variant_target_reads_the_target_suffix():
    assert (
        run_tests.variant_target("testcases/inline_asm_loop.aarch64_linux.lamina")
        == "aarch64_linux"
    )


def test_plain_testcase_is_not_a_variant():
    assert run_tests.variant_target("testcases/inline_asm_loop.lamina") is None


def test_discovery_never_returns_a_variant():
    discovered = run_tests.discover(str(ROOT / "testcases" / "*.lamina"))
    assert discovered
    assert all(run_tests.variant_target(path) is None for path in discovered)


@pytest.fixture
def variant_dir(request, monkeypatch):
    directory = SCRATCH / request.node.name
    directory.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(directory)
    (directory / "prog.lamina").write_text("")
    (directory / "prog.aarch64_linux.lamina").write_text("")
    (directory / "other.lamina").write_text("")
    return directory


def test_discover_skips_variant_files(variant_dir):
    assert run_tests.discover("*.lamina") == ["other.lamina", "prog.lamina"]


def test_testcase_source_picks_the_variant_for_its_target(variant_dir):
    assert run_tests.testcase_source(
        str(variant_dir / "prog.lamina"), "aarch64_linux"
    ) == str(variant_dir / "prog.aarch64_linux.lamina")


def test_testcase_source_falls_back_to_the_base_for_other_targets(variant_dir):
    prog = str(variant_dir / "prog.lamina")
    other = str(variant_dir / "other.lamina")
    assert run_tests.testcase_source(prog, "s390x_linux") == prog
    assert run_tests.testcase_source(other, "aarch64_linux") == other


@pytest.fixture
def runner(tmp_path, monkeypatch):
    monkeypatch.setattr(run_tests, "__file__", str(tmp_path / "run_tests.py"))
    monkeypatch.setattr(
        run_tests,
        "discover",
        lambda pattern: ["testcases/fake.lamina"] if "testcases" in pattern else [],
    )
    monkeypatch.setattr(run_tests, "load_expected_output", lambda *args, **kwargs: ["ok"])
    return monkeypatch


def test_failing_testcase_exits_zero_with_report_only(runner, capsys):
    runner.setattr(run_tests, "compile_and_run_test", lambda *args, **kwargs: (False, "boom"))
    run_tests.run_tests(report_only=True)
    assert "1 failed" in capsys.readouterr().out


def test_failing_testcase_exits_nonzero_without_report_only(runner):
    runner.setattr(run_tests, "compile_and_run_test", lambda *args, **kwargs: (False, "boom"))
    with pytest.raises(SystemExit) as exit_info:
        run_tests.run_tests()
    assert exit_info.value.code == 1


def test_missing_expected_output_counts_as_a_failure(runner, capsys):
    runner.setattr(run_tests, "load_expected_output", lambda *args, **kwargs: None)
    runner.setattr(
        run_tests,
        "compile_and_run_test",
        lambda *args, **kwargs: pytest.fail("a testcase without expectations was run"),
    )
    with pytest.raises(SystemExit) as exit_info:
        run_tests.run_tests()
    assert exit_info.value.code == 1
    out = capsys.readouterr().out
    assert re.search(r"FAILED fake\.lamina: No expected output found", out)
    assert "0/1 passed, 1 failed" in out
