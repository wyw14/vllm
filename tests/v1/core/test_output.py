# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy

import pytest
import torch

from vllm.multimodal.inputs import (
    MultiModalBatchedField,
    MultiModalFeatureSpec,
    MultiModalFieldElem,
    MultiModalKwargsItem,
    PlaceholderRange,
)
from vllm.multimodal.utils import strip_covered_mm_data
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)


def _create_new_requests_data(prompt_embeds: torch.Tensor | None) -> NewRequestData:
    return NewRequestData(
        req_id="test_req",
        prompt_token_ids=None,
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        block_ids=([],),
        num_computed_tokens=0,
        lora_request=None,
        prompt_embeds=prompt_embeds,
    )


def test_repr_with_none() -> None:
    """Test repr when prompt_embeds is None."""
    new_requests_data = _create_new_requests_data(None)

    assert "prompt_embeds_shape=None" in repr(new_requests_data)
    assert "prompt_embeds_shape=None" in new_requests_data.anon_repr()


def test_repr_with_multi_element_tensor() -> None:
    """Test repr when prompt_embeds is a multi-element tensor."""
    prompt_embeds = torch.randn(10, 768)
    new_requests_data = _create_new_requests_data(prompt_embeds)

    assert "prompt_embeds_shape=torch.Size([10, 768])" in repr(new_requests_data)
    assert "prompt_embeds_shape=torch.Size([10, 768])" in new_requests_data.anon_repr()


def _mm_feature(offset: int, length: int) -> MultiModalFeatureSpec:
    return MultiModalFeatureSpec(
        data=MultiModalKwargsItem.dummy(),
        mm_position=PlaceholderRange(offset=offset, length=length),
        identifier=f"hash_{offset}",
        modality="image",
    )


def test_strip_covered_mm_data() -> None:
    """Items fully inside the computed prefix lose their data; items touching
    the uncomputed region keep it; already-None data stays None."""
    from dataclasses import replace

    covered = _mm_feature(offset=0, length=100)
    boundary = _mm_feature(offset=150, length=100)  # ends exactly at 250
    uncovered = _mm_feature(offset=300, length=100)
    already_none = replace(_mm_feature(offset=100, length=50), data=None)

    stripped = strip_covered_mm_data(
        [covered, boundary, uncovered, already_none], num_computed_tokens=250
    )

    assert stripped[0].data is None  # fully covered -> stripped
    assert stripped[1].data is None  # span end == computed -> covered -> stripped
    assert stripped[2].data is not None  # extends past prefix -> kept
    assert stripped[3].data is None  # was already None
    # non-data fields are preserved
    assert stripped[0].identifier == covered.identifier
    assert stripped[0].mm_position == covered.mm_position
    # original list is not mutated
    assert covered.data is not None


def test_strip_covered_mm_data_zero_computed() -> None:
    """With no prefix hit nothing is stripped."""
    features = [_mm_feature(offset=0, length=100)]
    stripped = strip_covered_mm_data(features, num_computed_tokens=0)
    assert stripped[0].data is not None


def _mm_feature_mixed(offset: int, length: int) -> MultiModalFeatureSpec:
    data = MultiModalKwargsItem(
        {
            "pixel_values": MultiModalFieldElem(
                data=torch.empty(4), field=MultiModalBatchedField()
            ),
            "image_grid_thw": MultiModalFieldElem(
                data=torch.ones(1, 3, dtype=torch.long),
                field=MultiModalBatchedField(keep_on_cpu=True),
            ),
        }
    )
    return MultiModalFeatureSpec(
        data=data,
        mm_position=PlaceholderRange(offset=offset, length=length),
        identifier=f"hash_{offset}",
        modality="image",
    )


def test_strip_covered_mm_data_mrope() -> None:
    """For M-RoPE models, covered items keep their keep_on_cpu metadata fields
    (the worker needs them to compute positions); payload fields are dropped."""
    covered = _mm_feature_mixed(offset=0, length=100)
    uncovered = _mm_feature_mixed(offset=300, length=100)

    stripped = strip_covered_mm_data(
        [covered, uncovered], num_computed_tokens=250, uses_mrope=True
    )

    assert stripped[0].data is not None
    assert list(stripped[0].data.keys()) == ["image_grid_thw"]
    assert stripped[1].data is not None
    assert set(stripped[1].data.keys()) == {"pixel_values", "image_grid_thw"}
    # original list is not mutated
    assert set(covered.data.keys()) == {"pixel_values", "image_grid_thw"}


def test_strip_covered_mm_data_shm_address_item() -> None:
    """SHM address descriptors must survive stripping even when covered: the
    worker needs the address to resolve the payload and to acknowledge the
    sender's writer reference. Stripping it crashed the SHM receiver cache on
    the second identical request (vllm-project/vllm#54994)."""
    address_item = MultiModalKwargsItem(
        {
            "address": MultiModalFieldElem(data=4096, field=MultiModalBatchedField()),
            "monotonic_id": MultiModalFieldElem(data=7, field=MultiModalBatchedField()),
        }
    )
    feature = MultiModalFeatureSpec(
        data=address_item,
        mm_position=PlaceholderRange(offset=0, length=100),
        identifier="shm_item",
        modality="image",
    )

    stripped = strip_covered_mm_data([feature], num_computed_tokens=250)

    assert stripped[0].data is address_item


def _minimal_new_req_data(req_id: str) -> NewRequestData:
    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=None,
        mm_features=[],
        sampling_params=None,
        pooling_params=None,
        block_ids=([],),
        num_computed_tokens=0,
        lora_request=None,
    )


def _make_scheduler_output(
    new_req_ids: tuple[str, ...] = (),
    cached_req_ids: tuple[str, ...] = (),
    num_scheduled_tokens: dict[str, int] | None = None,
    total_num_scheduled_tokens: int = 0,
    preempted_req_ids: set[str] | None = None,
    resumed_req_ids: tuple[str, ...] = (),
) -> SchedulerOutput:
    return SchedulerOutput(
        scheduled_new_reqs=[_minimal_new_req_data(i) for i in new_req_ids],
        scheduled_cached_reqs=CachedRequestData(
            req_ids=list(cached_req_ids),
            resumed_req_ids=set(resumed_req_ids),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[None] * len(cached_req_ids),
            num_computed_tokens=[0] * len(cached_req_ids),
            num_output_tokens=[0] * len(cached_req_ids),
        ),
        num_scheduled_tokens=dict(num_scheduled_tokens or {}),
        total_num_scheduled_tokens=total_num_scheduled_tokens,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        preempted_req_ids=preempted_req_ids,
    )


def test_diagnose_scheduled_token_counts_empty_output() -> None:
    """An empty output has no counts and no findings."""
    report = SchedulerOutput.make_empty().diagnose_scheduled_token_counts()

    assert report.is_consistent
    assert report.declared_total == 0
    assert report.summed_total == 0
    assert report.total_mismatch == 0
    assert report.missing_new_req_ids == ()
    assert report.missing_cached_req_ids == ()
    assert report.zero_count_req_ids == ()
    assert dict(report.negative_counts) == {}
    assert report.duplicated_req_ids == ()
    assert "consistent" in report.format_report()


@pytest.mark.parametrize("resumed_as_new", [False, True])
def test_diagnose_scheduled_token_counts_preempted_requests(
    resumed_as_new: bool,
) -> None:
    """A preempted-and-resumed request is scheduled in one of the groups; a
    preempted-only request neither appears in a group nor needs a count."""
    if resumed_as_new:
        # v2 model runner: resumed preempted requests are re-sent as new.
        output = _make_scheduler_output(
            new_req_ids=("new-1", "resumed-1"),
            cached_req_ids=("run-1",),
            num_scheduled_tokens={"new-1": 2, "resumed-1": 3, "run-1": 1},
            total_num_scheduled_tokens=6,
            preempted_req_ids={"resumed-1", "evicted-1"},
        )
    else:
        # v1 model runner: resumed preempted requests travel as cached reqs.
        output = _make_scheduler_output(
            new_req_ids=("new-1",),
            cached_req_ids=("run-1", "resumed-1"),
            num_scheduled_tokens={"new-1": 2, "run-1": 1, "resumed-1": 3},
            total_num_scheduled_tokens=6,
            preempted_req_ids={"resumed-1", "evicted-1"},
            resumed_req_ids=("resumed-1",),
        )

    report = output.diagnose_scheduled_token_counts()

    assert report.is_consistent
    assert report.missing_new_req_ids == ()
    assert report.missing_cached_req_ids == ()
    assert report.duplicated_req_ids == ()
    # The preempted-only request must not be reported anywhere.
    assert "evicted-1" not in report.zero_count_req_ids
    assert "evicted-1" not in report.negative_counts


def test_diagnose_scheduled_token_counts_missing_distinguished_from_zero() -> None:
    """A missing key is reported, whereas an explicit 0 is a present count and
    must not be confused with a missing one; request order is preserved."""
    output = _make_scheduler_output(
        new_req_ids=("n1", "n2"),
        cached_req_ids=("c1", "c2", "c3"),
        num_scheduled_tokens={"n1": 4, "c1": 0, "c2": 3},
        total_num_scheduled_tokens=7,
    )

    report = output.diagnose_scheduled_token_counts()

    assert not report.is_consistent
    assert report.total_mismatch == 0
    assert report.missing_new_req_ids == ("n2",)
    assert report.missing_cached_req_ids == ("c3",)
    assert report.zero_count_req_ids == ("c1",)
    assert report.summed_total == 7
    assert dict(report.negative_counts) == {}
    text = report.format_report()
    assert "new requests missing a count: ['n2']" in text
    assert "cached requests missing a count: ['c3']" in text
    assert "'c1'" in text


def test_diagnose_scheduled_token_counts_negative_counts() -> None:
    """Negative counts are flagged in scheduled order, including ids that only
    exist in the counts mapping."""
    output = _make_scheduler_output(
        new_req_ids=("a",),
        cached_req_ids=("b", "c"),
        num_scheduled_tokens={"a": 1, "b": -2, "c": 3, "ghost": -5},
        total_num_scheduled_tokens=-3,
    )

    report = output.diagnose_scheduled_token_counts()

    assert not report.is_consistent
    # Scheduled ids keep their new-then-cached order; the unscheduled id is
    # appended afterwards rather than dropping its invalid count.
    assert list(report.negative_counts.items()) == [("b", -2), ("ghost", -5)]
    assert report.summed_total == -3
    assert report.missing_new_req_ids == ()
    assert report.missing_cached_req_ids == ()


def test_diagnose_scheduled_token_counts_duplicated_across_groups() -> None:
    """A request present in both the new and cached groups is reported in new
    request order, while its count still counts only once."""
    output = _make_scheduler_output(
        new_req_ids=("x", "y"),
        cached_req_ids=("x", "z"),
        num_scheduled_tokens={"x": 1, "y": 1, "z": 1},
        total_num_scheduled_tokens=3,
    )

    report = output.diagnose_scheduled_token_counts()

    assert not report.is_consistent
    assert report.duplicated_req_ids == ("x",)
    assert report.missing_new_req_ids == ()
    assert report.missing_cached_req_ids == ()
    assert report.summed_total == 3


@pytest.mark.parametrize("declared_total", [6, 4])
def test_diagnose_scheduled_token_counts_total_mismatch(
    declared_total: int,
) -> None:
    """A declared total that differs from the per-request sum is reported with
    a signed difference."""
    output = _make_scheduler_output(
        new_req_ids=("a",),
        cached_req_ids=("b",),
        num_scheduled_tokens={"a": 2, "b": 3},
        total_num_scheduled_tokens=declared_total,
    )

    report = output.diagnose_scheduled_token_counts()

    assert not report.is_consistent
    assert report.summed_total == 5
    assert report.total_mismatch == declared_total - 5
    assert report.missing_new_req_ids == ()
    assert report.missing_cached_req_ids == ()
    text = report.format_report()
    assert "inconsistent" in text
    assert f"difference {report.total_mismatch:+d}" in text


def test_diagnose_scheduled_token_counts_does_not_mutate_output() -> None:
    """The diagnostic is read-only: the scheduler output is unchanged."""
    output = _make_scheduler_output(
        new_req_ids=("n1",),
        cached_req_ids=("c1",),
        num_scheduled_tokens={"c1": -1},
        total_num_scheduled_tokens=0,
        preempted_req_ids={"p1"},
    )
    snapshot = copy.deepcopy(output)

    output.diagnose_scheduled_token_counts()

    assert output == snapshot
