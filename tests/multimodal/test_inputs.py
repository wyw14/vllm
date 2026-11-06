# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
import torch

from vllm.multimodal.inputs import (
    PlaceholderRange,
    PlaceholderRangeIssueKind,
    diagnose_placeholder_ranges,
)


def _issue_keys(issues):
    """Compact (range index, issue kind, other range index) per issue."""
    return [(issue.index, issue.kind, issue.other_index) for issue in issues]


@pytest.mark.parametrize(
    "is_embed,expected",
    [
        (None, 5),
        (torch.tensor([True, True, True, True, True]), 5),
        (torch.tensor([False, False, False, False, False]), 0),
        (torch.tensor([True, False, True, False, True]), 3),
        (torch.tensor([True]), 1),
    ],
)
def test_placeholder_range_get_num_embeds(is_embed, expected):
    length = len(is_embed) if is_embed is not None else 5
    pr = PlaceholderRange(offset=0, length=length, is_embed=is_embed)
    assert pr.get_num_embeds() == expected


@pytest.mark.parametrize(
    "is_embed,expected",
    [
        (None, None),
        (torch.tensor([False, True, False, True, True]), [0, 1, 1, 2, 3]),
        (torch.tensor([True, True, True]), [1, 2, 3]),
    ],
)
def test_placeholder_range_embeds_cumsum(is_embed, expected):
    length = len(is_embed) if is_embed is not None else 5
    pr = PlaceholderRange(offset=0, length=length, is_embed=is_embed)

    if expected is None:
        assert pr.embeds_cumsum is None
        return

    assert pr.embeds_cumsum == expected
    # cached_property should return the same object on repeated access
    assert pr.embeds_cumsum is pr.embeds_cumsum


def test_diagnose_placeholder_ranges_empty_list():
    assert diagnose_placeholder_ranges(8, []) == []


@pytest.mark.parametrize(
    "token_total_length,ranges",
    [
        # single range fitting exactly, half-open end equals total length
        (4, [PlaceholderRange(offset=0, length=4)]),
        # adjacent ranges share only the boundary: [0, 4) and [4, 8)
        (
            8,
            [
                PlaceholderRange(offset=0, length=4),
                PlaceholderRange(offset=4, length=4),
            ],
        ),
        # zero-length ranges never overlap, including one at the end
        (
            4,
            [
                PlaceholderRange(offset=0, length=4),
                PlaceholderRange(offset=2, length=0),
                PlaceholderRange(offset=4, length=0),
            ],
        ),
        # is_embed masks matching their range lengths
        (
            8,
            [
                PlaceholderRange(
                    offset=0,
                    length=2,
                    is_embed=torch.tensor([True, False]),
                ),
                PlaceholderRange(
                    offset=2,
                    length=0,
                    is_embed=torch.tensor([], dtype=torch.bool),
                ),
            ],
        ),
    ],
)
def test_diagnose_placeholder_ranges_valid(token_total_length, ranges):
    assert diagnose_placeholder_ranges(token_total_length, ranges) == []


def test_diagnose_placeholder_ranges_negative_offset():
    # Ends inside the prompt ([-1, 1)), so the only problem is the offset.
    issues = diagnose_placeholder_ranges(
        4, [PlaceholderRange(offset=0, length=2), PlaceholderRange(-1, 2)]
    )
    assert _issue_keys(issues) == [(1, PlaceholderRangeIssueKind.NEGATIVE_OFFSET, None)]
    # Negative-offset ranges do not participate in overlap detection.
    assert "overlap" not in issues[0].message


def test_diagnose_placeholder_ranges_negative_length():
    issues = diagnose_placeholder_ranges(4, [PlaceholderRange(offset=1, length=-2)])
    assert _issue_keys(issues) == [(0, PlaceholderRangeIssueKind.NEGATIVE_LENGTH, None)]


def test_diagnose_placeholder_ranges_out_of_bounds():
    issues = diagnose_placeholder_ranges(
        8,
        [
            PlaceholderRange(offset=6, length=4),  # [6, 10) exceeds 8
            PlaceholderRange(offset=9, length=0),  # empty past the end
        ],
    )
    assert _issue_keys(issues) == [
        (0, PlaceholderRangeIssueKind.OUT_OF_BOUNDS, None),
        (1, PlaceholderRangeIssueKind.OUT_OF_BOUNDS, None),
    ]


def test_diagnose_placeholder_ranges_partial_overlap():
    # Partial overlap: [0, 5) vs [4, 8); adjacent pair has no overlap.
    issues = diagnose_placeholder_ranges(
        8,
        [
            PlaceholderRange(offset=0, length=5),
            PlaceholderRange(offset=4, length=4),
            PlaceholderRange(offset=8, length=2),
        ],
    )
    assert _issue_keys(issues) == [
        (1, PlaceholderRangeIssueKind.OVERLAP, 0),
        (2, PlaceholderRangeIssueKind.OUT_OF_BOUNDS, None),
    ]


@pytest.mark.parametrize(
    "ranges",
    [
        # later range fully contained in the earlier one
        [
            PlaceholderRange(offset=0, length=10),
            PlaceholderRange(offset=2, length=2),
        ],
        # earlier range fully contained in the later one
        [
            PlaceholderRange(offset=2, length=2),
            PlaceholderRange(offset=0, length=10),
        ],
    ],
)
def test_diagnose_placeholder_ranges_containment(ranges):
    issues = diagnose_placeholder_ranges(10, ranges)
    assert _issue_keys(issues) == [(1, PlaceholderRangeIssueKind.OVERLAP, 0)]
    assert str(0) in issues[0].message and str(1) in issues[0].message


@pytest.mark.parametrize(
    "ranges",
    [
        # zero-length interval inside the earlier range
        [
            PlaceholderRange(offset=0, length=10),
            PlaceholderRange(offset=2, length=0),
        ],
        # zero-length interval before the non-empty one
        [
            PlaceholderRange(offset=2, length=0),
            PlaceholderRange(offset=0, length=10),
        ],
    ],
)
def test_diagnose_placeholder_ranges_empty_interval_inside_range(ranges):
    # A zero-length interval never overlaps another interval.
    assert diagnose_placeholder_ranges(10, ranges) == []


@pytest.mark.parametrize(
    "length,mask,expected_kind",
    [
        (
            3,
            torch.tensor([True, False, True, True]),
            PlaceholderRangeIssueKind.MASK_LENGTH_MISMATCH,
        ),
        (0, torch.tensor([True]), PlaceholderRangeIssueKind.MASK_LENGTH_MISMATCH),
        (3, torch.tensor([True, False, True]), None),
        (0, torch.tensor([], dtype=torch.bool), None),
    ],
)
def test_diagnose_placeholder_ranges_mask_length(length, mask, expected_kind):
    issues = diagnose_placeholder_ranges(
        4, [PlaceholderRange(offset=0, length=length, is_embed=mask)]
    )
    if expected_kind is None:
        assert issues == []
    else:
        assert _issue_keys(issues) == [(0, expected_kind, None)]


def test_diagnose_placeholder_ranges_reports_all_in_stable_order():
    # idx 0: [0, 6)
    # idx 1: [4, 10) -- overlaps idx 0
    # idx 2: [2, 8)  -- overlaps idx 0 then idx 1 (input order)
    # idx 3: negative offset -- no overlap checks, not out of bounds
    # idx 4: [18, 22) -- out of bounds (total 20)
    # idx 5: empty [19, 19) -- no problem, does not overlap idx 4
    ranges = [
        PlaceholderRange(offset=0, length=6),
        PlaceholderRange(offset=4, length=6),
        PlaceholderRange(offset=2, length=6),
        PlaceholderRange(offset=-2, length=3),
        PlaceholderRange(offset=18, length=4),
        PlaceholderRange(offset=19, length=0),
    ]
    issues = diagnose_placeholder_ranges(20, ranges)
    assert _issue_keys(issues) == [
        (1, PlaceholderRangeIssueKind.OVERLAP, 0),
        (2, PlaceholderRangeIssueKind.OVERLAP, 0),
        (2, PlaceholderRangeIssueKind.OVERLAP, 1),
        (3, PlaceholderRangeIssueKind.NEGATIVE_OFFSET, None),
        (4, PlaceholderRangeIssueKind.OUT_OF_BOUNDS, None),
    ]


def test_diagnose_placeholder_ranges_does_not_touch_inputs():
    mask = torch.tensor([True, False])
    ranges = [
        PlaceholderRange(offset=0, length=2, is_embed=mask),
        PlaceholderRange(offset=3, length=1),
    ]
    snapshot = [
        (r.offset, r.length, None if r.is_embed is None else r.is_embed.clone())
        for r in ranges
    ]

    first = diagnose_placeholder_ranges(4, tuple(ranges))
    second = diagnose_placeholder_ranges(4, ranges)

    # Deterministic and independent of the input container type.
    assert first == second
    # Request data is neither mutated nor read beyond mask metadata.
    for rng, (offset, length, mask_before) in zip(ranges, snapshot, strict=True):
        assert (rng.offset, rng.length) == (offset, length)
        if mask_before is None:
            assert rng.is_embed is None
        else:
            assert torch.equal(rng.is_embed, mask_before)
