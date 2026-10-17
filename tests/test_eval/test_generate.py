"""Tests for eval/mbench_baseline/generate.py — pure CSV/adapter logic, no GPU/network."""

import csv

from eval.mbench_baseline import generate as G


def _write(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def test_read_generated_tolerates_missing_column(tmp_path):
    p = tmp_path / "base.csv"
    _write(
        p,
        ["idx", "context", "qwen3_4b_instruct_2507"],
        [{"idx": "0", "context": "c0", "qwen3_4b_instruct_2507": "r0"}],
    )
    got = G.read_generated(str(p), col="qwen3_4b_sft5")
    assert got == {0: ("c0", "")}


def test_read_generated_reads_matching_column(tmp_path):
    p = tmp_path / "sft5.csv"
    _write(
        p,
        ["idx", "context", "qwen3_4b_sft5"],
        [{"idx": "3", "context": "c3", "qwen3_4b_sft5": "r3"}],
    )
    assert G.read_generated(str(p), col="qwen3_4b_sft5") == {3: ("c3", "r3")}


def test_completed_keeps_only_nonempty_responses(tmp_path):
    p = tmp_path / "o.csv"
    _write(
        p,
        ["idx", "context", "qwen3_4b_sft5"],
        [
            {"idx": "0", "context": "c0", "qwen3_4b_sft5": "r0"},
            {"idx": "1", "context": "c1", "qwen3_4b_sft5": "  "},
            {"idx": "2", "context": "c2", "qwen3_4b_sft5": ""},
        ],
    )
    assert G.completed(str(p), col="qwen3_4b_sft5") == {0}


def test_find_adapter_explicit_none_and_hit(tmp_path):
    assert G.find_adapter("/nope", _roots=[str(tmp_path)]) is None
    d = tmp_path / "x" / "y" / "adapter"
    d.mkdir(parents=True)
    (d / "adapter_config.json").write_text("{}", encoding="utf-8")
    assert G.find_adapter(None, _roots=[str(tmp_path)]) == str(d)


def test_strip_to_generated_drops_left_padded_prompt():
    import torch

    pad_width = 6  # common padded input width
    out = torch.tensor(
        [
            [0, 0, 1, 2, 3, 4, 9, 9],  # 2 pads + 4 real + 2 generated
            [0, 5, 6, 7, 8, 9, 8, 8],  # 1 pad + 5 real + 2 generated
        ]
    )
    assert G.strip_to_generated(out, pad_width).tolist() == [[9, 9], [8, 8]]
    # the old per-row-length slice leaked prompt tokens for padded rows:
    assert out[0][4:].tolist() == [3, 4, 9, 9]
