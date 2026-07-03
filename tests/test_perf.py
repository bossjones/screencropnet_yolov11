"""Unit tests for the torch-free profiling core in ``screencropnet_yolo.perf``.

These cover the pure building blocks the ``perf`` skill relies on: pytest
argument-vector construction (with the mandatory coverage/timeout strip),
profiler command builders, report naming, profiler-output parsing, and markdown
rendering. No subprocess, no torch, no scalene here — those are exercised by the
skill's real profiling runs.
"""

from __future__ import annotations

import pytest

from screencropnet_yolo import perf


class TestBuildPytestArgv:
    def test_always_strips_coverage_and_timeout(self) -> None:
        argv = perf.build_pytest_argv("tests/test_inference_e2e.py")
        # Coverage's sys.settrace and the 30s thread-timeout both corrupt a
        # profiled run, so every invocation must neutralize them.
        assert "no:cov" in argv
        assert "addopts=" in argv
        assert "--timeout=0" in argv
        # Clearing addopts also drops the repo's --import-mode=importlib, which
        # it needs to avoid duplicate-basename collection errors — restore it.
        assert "--import-mode=importlib" in argv

    def test_selects_the_requested_marker(self) -> None:
        argv = perf.build_pytest_argv("tests/x.py", marker="pytorch_profile")
        i = argv.index("-m")
        assert argv[i + 1] == "pytorch_profile"

    def test_carries_the_target_and_extra_args(self) -> None:
        argv = perf.build_pytest_argv("tests/x.py::test_y", extra=["-s"])
        assert "tests/x.py::test_y" in argv
        assert "-s" in argv


class TestScaleneCommand:
    def test_runs_scalene_ephemerally_with_uv(self) -> None:
        cmd = perf.scalene_command("/skills/_pytest_runner.py", "tests/x.py", "/tmp/out.json")
        joined = " ".join(cmd)
        assert cmd[:4] == ["uv", "run", "--with", "scalene"]
        # Scalene 2.3 uses `scalene run -o OUTFILE PROGRAM --- prog-args`.
        assert "run" in cmd
        assert cmd[cmd.index("-o") + 1] == "/tmp/out.json"
        assert "/skills/_pytest_runner.py" in cmd
        # The target + pytest args are the program's args, after the separator.
        sep = cmd.index("---")
        assert "tests/x.py" in cmd[sep:]
        # The profiled pytest run still strips coverage + timeout.
        assert "no:cov" in joined
        assert "--timeout=0" in joined


class TestTorchPytestCommand:
    def test_uses_worker_group_and_marker(self) -> None:
        cmd = perf.torch_pytest_command("tests/x.py")
        assert cmd[:4] == ["uv", "run", "--group", "worker"]
        assert "pytest" in cmd
        i = cmd.index("-m")
        assert cmd[i + 1] == "pytorch_profile"
        assert "no:cov" in cmd


class TestReportNaming:
    def test_label_from_a_file_target(self) -> None:
        assert perf.label_from_target("tests/test_inference_e2e.py") == "test-inference-e2e"

    def test_label_from_a_node_id(self) -> None:
        assert perf.label_from_target("tests/x.py::test_predicts_fast") == "test-predicts-fast"

    def test_report_filename_prefixes_the_date(self) -> None:
        name = perf.report_filename("2026-07-02", "test-inference-e2e")
        assert name == "2026-07-02-test-inference-e2e.md"


class TestScaleneRows:
    def test_ranks_lines_by_total_cpu_percent(self) -> None:
        data = {
            "files": {
                "src/screencropnet_yolo/inference.py": {
                    "lines": [
                        {
                            "lineno": 10,
                            "line": "cheap()",
                            "n_cpu_percent_python": 1.0,
                            "n_cpu_percent_c": 0.0,
                            "n_peak_mb": 0.0,
                            "n_gpu_percent": 0.0,
                        },
                        {
                            "lineno": 20,
                            "line": "self.model.predict(x)",
                            "n_cpu_percent_python": 5.0,
                            "n_cpu_percent_c": 40.0,
                            "n_peak_mb": 128.0,
                            "n_gpu_percent": 0.0,
                        },
                    ]
                }
            }
        }
        rows = perf.scalene_rows(data, top=5)
        assert rows[0].lineno == 20
        assert rows[0].cpu_pct == pytest.approx(45.0)
        assert rows[0].peak_mb == pytest.approx(128.0)

    def test_drops_lines_with_no_attributed_cost(self) -> None:
        data = {
            "files": {
                "a.py": {
                    "lines": [
                        {
                            "lineno": 1,
                            "line": "# comment",
                            "n_cpu_percent_python": 0.0,
                            "n_cpu_percent_c": 0.0,
                            "n_peak_mb": 0.0,
                            "n_gpu_percent": 0.0,
                        },
                        {
                            "lineno": 2,
                            "line": "work()",
                            "n_cpu_percent_python": 3.0,
                            "n_cpu_percent_c": 0.0,
                            "n_peak_mb": 0.0,
                            "n_gpu_percent": 0.0,
                        },
                    ]
                }
            }
        }
        rows = perf.scalene_rows(data)
        assert [r.lineno for r in rows] == [2]

    def test_top_bounds_the_result(self) -> None:
        lines = [
            {
                "lineno": n,
                "line": "x",
                "n_cpu_percent_python": float(n),
                "n_cpu_percent_c": 0.0,
                "n_peak_mb": 0.0,
                "n_gpu_percent": 0.0,
            }
            for n in range(1, 11)
        ]
        data = {"files": {"a.py": {"lines": lines}}}
        assert len(perf.scalene_rows(data, top=3)) == 3


class TestTorchRows:
    def test_ranks_ops_by_self_cpu_time_without_cuda(self) -> None:
        events = [
            {
                "name": "aten::conv2d",
                "count": 30,
                "self_cpu_time_total": 900.0,
                "cpu_time_total": 1200.0,
                "self_cuda_time_total": 0.0,
                "cpu_memory_usage": 4096,
                "cuda_memory_usage": 0,
            },
            {
                "name": "aten::add",
                "count": 100,
                "self_cpu_time_total": 50.0,
                "cpu_time_total": 60.0,
                "self_cuda_time_total": 0.0,
                "cpu_memory_usage": 0,
                "cuda_memory_usage": 0,
            },
        ]
        rows = perf.torch_rows(events, top=5)
        assert rows[0].name == "aten::conv2d"
        assert rows[0].calls == 30
        assert rows[0].self_cpu_us == pytest.approx(900.0)


class TestRendering:
    def test_scalene_markdown_has_title_command_and_candidates(self) -> None:
        rows = [perf.ScaleneRow("inference.py", 20, "predict()", 45.0, 5.0, 40.0, 128.0, 0.0)]
        md = perf.render_scalene_markdown(
            target="tests/test_inference_e2e.py",
            command=["uv", "run", "--with", "scalene"],
            rows=rows,
            date_str="2026-07-02",
        )
        assert md.startswith("# Scalene profile")
        assert "tests/test_inference_e2e.py" in md
        assert "## Optimization candidates" in md
        assert "predict()" in md

    def test_torch_markdown_notes_cuda_availability(self) -> None:
        rows = [perf.TorchRow("aten::conv2d", 30, 900.0, 1200.0, 0.0, 4096, 0)]
        md = perf.render_torch_markdown(
            target="tests/test_inference_e2e.py",
            command=["uv", "run", "--group", "worker"],
            rows=rows,
            date_str="2026-07-02",
            trace_path="docs/perf/pytorch_profiler/trace.json",
            cuda_available=False,
        )
        assert md.startswith("# PyTorch Profiler")
        assert "aten::conv2d" in md
        assert "CUDA" in md
        assert "## Optimization candidates" in md
