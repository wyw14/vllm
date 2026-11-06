# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from vllm import SamplingParams
from vllm.config.diffusion import DiffusionConfig
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import (
    _MAX_TEMP,
    _SAMPLING_EPS,
    preview_sampling_params_normalization,
)
from vllm.utils.diffusion import validate_diffusion_sampling_params
from vllm.v1.engine.input_processor import InputProcessor


def preview(**kwargs):
    return preview_sampling_params_normalization(**kwargs)


@dataclass
class MockModelConfig:
    is_diffusion: bool = False
    max_logprobs: int = 20
    logits_processors: list | None = None
    return_sampling_mask: bool = False

    def get_vocab_size(self) -> int:
        return 1024


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": 0.7},
        {"temperature": 0.0},
        {"min_p": 0.1},
        {"seed": 42},
        {"min_tokens": 5},
        {"logit_bias": {0: 1.0}},
        {"bad_words": ["foo"]},
        {"allowed_token_ids": [0, 1]},
    ],
)
def test_diffusion_rejects_unsupported_params(kwargs: dict):
    params = SamplingParams(**kwargs)
    with pytest.raises(VLLMValidationError, match="not yet supported with diffusion"):
        params.verify(MockModelConfig(is_diffusion=True), None, None, None)


def test_diffusion_accepts_default_params():
    SamplingParams().verify(MockModelConfig(is_diffusion=True), None, None, None)


def test_diffusion_accepts_top_k_top_p():
    params = SamplingParams(top_p=0.9, top_k=10)
    params.verify(MockModelConfig(is_diffusion=True), None, None, None)


def test_non_diffusion_models_unaffected():
    params = SamplingParams(temperature=0.7, top_k=10, seed=42)
    params.verify(MockModelConfig(), None, None, None)


def test_verify_leaves_logits_processors_to_admission():
    """verify() is runner-agnostic; LP validation lives in the admission
    layer, so an unimportable FQCN must not fail verify()."""
    SamplingParams().verify(
        MockModelConfig(logits_processors=["no.such:Cls"]), None, None, None
    )


def _verify_diffusion(params: SamplingParams, canvas_length: int | None = None):
    model_config = MockModelConfig(is_diffusion=True)
    params.verify(model_config, None, None, None)
    validate_diffusion_sampling_params(
        params,
        canvas_length=canvas_length,
        vocab_size=model_config.get_vocab_size(),
    )


def test_diffusion_extra_args_are_validated_without_a_served_canvas():
    # No --diffusion-config: the canvas is unknown, the ids are still checked
    # and a read-only request is still normalised.
    params = SamplingParams(
        max_tokens=64,
        extra_args={"diffusion_seed_canvas": [0, 1], "diffusion_read_only": True},
    )
    _verify_diffusion(params, canvas_length=None)
    assert params.ignore_eos is True

    bad = SamplingParams(extra_args={"diffusion_seed_canvas": [0, 10**9]})
    with pytest.raises(VLLMValidationError, match="ids must be in"):
        _verify_diffusion(bad, canvas_length=None)


@pytest.mark.parametrize("async_scheduling", [False, True])
@pytest.mark.parametrize(
    "extra_args",
    [
        {},
        {"diffusion_canvas_length": None},
        {"diffusion_canvas_length": 4},
        {"diffusion_canvas_length": 8},
    ],
)
def test_diffusion_canvas_width_is_accepted_with_either_scheduler(
    async_scheduling, extra_args
):
    processor = SimpleNamespace(
        model_config=MockModelConfig(is_diffusion=True),
        vllm_config=SimpleNamespace(
            scheduler_config=SimpleNamespace(async_scheduling=async_scheduling)
        ),
        speculative_config=None,
        structured_outputs_config=None,
        diffusion_config=DiffusionConfig(canvas_length=8),
        tokenizer=None,
        validate_logits_processors_params=lambda params: None,
        resolve_watermarking=lambda params: False,
    )
    params = SamplingParams(extra_args=extra_args)
    InputProcessor._validate_params(processor, params, ("generate",))


@pytest.mark.parametrize(
    "extra_args, match",
    [
        ({"diffusion_seed_canvas": "abc"}, "list of token ids"),
        ({"diffusion_seed_canvas": [1, 2.5]}, "list of token ids"),
        ({"diffusion_seed_canvas": [1, True]}, "list of token ids"),
        ({"diffusion_seed_canvas": [0, 1024]}, r"in \[0, 1024\)"),
        ({"diffusion_seed_canvas": [0, -1]}, r"in \[0, 1024\)"),
        ({"diffusion_max_steps": 0}, "positive integer"),
        ({"diffusion_max_steps": "1"}, "positive integer"),
        ({"diffusion_max_steps": True}, "positive integer"),
        ({"diffusion_read_only": "yes"}, "boolean"),
        ({"diffusion_read_only": 2}, "boolean"),
        ({"diffusion_pinned": "0,1"}, "list of canvas positions"),
        ({"diffusion_pinned": [0, True]}, "list of canvas positions"),
        ({"diffusion_pinned": [0]}, "needs a diffusion_seed_canvas"),
    ],
)
def test_diffusion_rejects_bad_extra_args(extra_args: dict, match: str):
    with pytest.raises(VLLMValidationError, match=match):
        _verify_diffusion(SamplingParams(extra_args=extra_args))


@pytest.mark.parametrize("flag", [True, 1])
def test_diffusion_constrained_needs_logprob_token_ids(flag):
    bad = SamplingParams(extra_args={"diffusion_constrained": flag})
    with pytest.raises(VLLMValidationError, match="needs logprob_token_ids"):
        _verify_diffusion(bad)

    ok = SamplingParams(
        logprob_token_ids=[3, 5], extra_args={"diffusion_constrained": flag}
    )
    _verify_diffusion(ok)

    # An unset or false flag needs no ids.
    _verify_diffusion(SamplingParams(extra_args={"diffusion_constrained": False}))
    _verify_diffusion(SamplingParams(extra_args={"diffusion_constrained": 0}))


@pytest.mark.parametrize("value", ["yes", 2, [1]])
def test_diffusion_constrained_must_be_a_bool(value):
    params = SamplingParams(extra_args={"diffusion_constrained": value})
    with pytest.raises(VLLMValidationError, match="must be a boolean"):
        _verify_diffusion(params)


def test_diffusion_seed_canvas_must_fill_the_canvas():
    params = SamplingParams(extra_args={"diffusion_seed_canvas": [0] * 7})
    with pytest.raises(VLLMValidationError, match="exactly 8 ids, got 7"):
        _verify_diffusion(params, canvas_length=8)
    # Without the served diffusion config the canvas length is unknown.
    _verify_diffusion(params)


@pytest.mark.parametrize("max_tokens, expected", [(100, 8), (5, 5), (None, 8)])
@pytest.mark.parametrize("flag", [True, 1])
def test_diffusion_read_only_ends_after_one_canvas(max_tokens, expected, flag):
    params = SamplingParams(
        max_tokens=max_tokens, extra_args={"diffusion_read_only": flag}
    )
    _verify_diffusion(params, canvas_length=8)
    assert params.max_tokens == expected
    assert params.ignore_eos


@pytest.mark.parametrize(
    "width, match",
    [
        (0, "positive integer"),
        ("4", "positive integer"),
        (True, "positive integer"),
        (9, "no larger"),
    ],
)
def test_diffusion_rejects_bad_canvas_length(width, match):
    with pytest.raises(VLLMValidationError, match=match):
        _verify_diffusion(
            SamplingParams(extra_args={"diffusion_canvas_length": width}),
            canvas_length=8,
        )


def test_diffusion_canvas_length_sizes_the_seed_and_the_read():
    params = SamplingParams(
        max_tokens=100,
        extra_args={
            "diffusion_canvas_length": 4,
            "diffusion_seed_canvas": [1, 2, 3, 4],
            "diffusion_read_only": True,
        },
    )
    _verify_diffusion(params, canvas_length=8)
    assert params.max_tokens == 4

    params = SamplingParams(
        extra_args={"diffusion_canvas_length": 4, "diffusion_seed_canvas": [0] * 8}
    )
    with pytest.raises(VLLMValidationError, match="exactly 4 ids, got 8"):
        _verify_diffusion(params, canvas_length=8)


def test_diffusion_pinned_positions_stay_inside_the_canvas():
    seed = [0] * 8
    with pytest.raises(VLLMValidationError, match="inside the canvas"):
        _verify_diffusion(
            SamplingParams(
                extra_args={"diffusion_seed_canvas": seed, "diffusion_pinned": [7, 8]}
            ),
            canvas_length=8,
        )
    with pytest.raises(VLLMValidationError, match="inside the canvas"):
        _verify_diffusion(
            SamplingParams(
                extra_args={"diffusion_seed_canvas": seed, "diffusion_pinned": [-1]}
            )
        )
    # A narrower request canvas bounds the positions.
    with pytest.raises(VLLMValidationError, match="inside the canvas"):
        _verify_diffusion(
            SamplingParams(
                extra_args={
                    "diffusion_canvas_length": 4,
                    "diffusion_seed_canvas": seed[:4],
                    "diffusion_pinned": [4],
                }
            ),
            canvas_length=8,
        )


def test_diffusion_accepts_extra_args():
    params = SamplingParams(
        extra_args={
            "diffusion_seed_canvas": list(range(8)),
            "diffusion_pinned": [0, 1, 7],
            "diffusion_max_steps": 4,
            "diffusion_read_only": True,
        }
    )
    _verify_diffusion(params, canvas_length=8)


@pytest.mark.parametrize("value", [-(2**63) - 1, 2**64])
def test_extra_args_rejects_nested_integer_overflow(value):
    """Reject extension values before they reach the engine transport."""
    with pytest.raises(VLLMValidationError, match="extra_args integers"):
        SamplingParams(extra_args={"ec_transfer_params": {"nested": [{"x": value}]}})


@pytest.mark.parametrize("value", [-(2**63), 2**63 - 1, 2**63, 2**64 - 1, True])
def test_extra_args_accepts_messagepack_integer_boundaries(value):
    extra_args = {"kv_transfer_params": {"nested": [{"x": value}]}}
    assert SamplingParams(extra_args=extra_args).extra_args == extra_args


def test_extra_args_preserves_custom_objects_and_shared_containers():
    custom = object()
    shared = [custom, (None, "value", 1.5)]
    extra_args = {"first": shared, "second": shared}
    params = SamplingParams(extra_args=extra_args)
    assert params.extra_args["first"][0] is custom
    assert params.extra_args["first"] is params.extra_args["second"]


def test_preview_with_empty_args_reports_nothing():
    assert preview() == []
    # Explicit empty containers are already in their normalized form.
    assert preview(stop=[]) == []
    assert preview(stop_token_ids=[]) == []
    assert preview(bad_words=[]) == []


def test_preview_wraps_stop_string_without_echoing_text():
    secret = "SECRET-STOP-VALUE"
    changes = preview(stop=secret)
    assert [change.field for change in changes] == ["stop"]
    assert "list" in changes[0].reason
    assert secret not in repr(changes)
    assert SamplingParams(stop=secret).stop == [secret]
    # An already-normalized list is left alone.
    assert preview(stop=["a", "b"]) == []
    # An explicit None is normalized to an empty list.
    (change,) = preview(stop=None)
    assert change.field == "stop"


def test_preview_deduplicates_stop_token_ids_without_echoing_ids():
    ids = [11, 22, 11, 22, 11]
    changes = preview(stop_token_ids=ids)
    assert [change.field for change in changes] == ["stop_token_ids"]
    assert "duplicate" in changes[0].reason
    assert "11" not in repr(changes)
    assert "22" not in repr(changes)
    # The preview agrees with direct construction.
    assert SamplingParams(stop_token_ids=ids).stop_token_ids == [11, 22]
    # Unique values and the original input are left untouched.
    assert preview(stop_token_ids=[1, 2]) == []
    assert ids == [11, 22, 11, 22, 11]


def test_preview_deduplicates_bad_words():
    secret = "SECRET-BAD-WORD"
    changes = preview(bad_words=[secret, secret])
    assert [change.field for change in changes] == ["bad_words"]
    assert "duplicate" in changes[0].reason
    assert secret not in repr(changes)


def test_preview_clears_seed_minus_one():
    changes = preview(seed=-1)
    assert [change.field for change in changes] == ["seed"]
    assert SamplingParams(seed=-1).seed is None
    assert preview(seed=42) == []


def test_preview_greedy_resets_sampling_options():
    changes = preview(temperature=0.0, top_p=0.5, top_k=10, min_p=0.2)
    assert [change.field for change in changes] == ["top_p", "top_k", "min_p"]
    assert all("greedy" in change.reason for change in changes)
    # temperature itself is not rewritten at exactly zero.
    assert SamplingParams(temperature=0.0).temperature == 0.0
    # Defaults already match the greedy reset, so nothing is reported.
    assert preview(temperature=0.0) == []
    # Only non-default options surface.
    (change,) = preview(temperature=0.0, top_k=10)
    assert change.field == "top_k"


@pytest.mark.parametrize(
    ("temperature", "expected_fields"),
    [
        (0.0, []),  # exact zero selects greedy without a rewrite
        (_SAMPLING_EPS / 2, ["temperature"]),  # raised to the floor
        (_SAMPLING_EPS, ["temperature"]),  # below the floor -> raised
        (_MAX_TEMP / 2, ["temperature"]),
        (_MAX_TEMP - 1e-12, ["temperature"]),
        (_MAX_TEMP, []),  # exactly the floor is untouched
        (0.5, []),
        (1.0, []),
        (2.0, []),  # upper bound is inclusive
    ],
)
def test_preview_temperature_boundaries(temperature, expected_fields):
    changes = preview(temperature=temperature)
    assert [change.field for change in changes] == expected_fields
    params = SamplingParams(temperature=temperature)
    if expected_fields:
        assert params.temperature == _MAX_TEMP
    else:
        assert params.temperature == temperature


def test_preview_reports_changes_in_post_init_order():
    changes = preview(
        temperature=0.0,
        seed=-1,
        stop="x",
        stop_token_ids=[1, 1],
        top_p=0.5,
    )
    assert [change.field for change in changes] == [
        "seed",
        "stop",
        "stop_token_ids",
        "top_p",
    ]


def test_preview_does_not_mutate_caller_input():
    ids = [3, 3, 7]
    words = ["dup", "dup"]
    changes = preview(
        stop=["keep"],
        stop_token_ids=ids,
        bad_words=words,
        seed=-1,
        temperature=0.0,
        top_p=0.5,
    )
    assert {change.field for change in changes} == {
        "stop_token_ids",
        "bad_words",
        "seed",
        "top_p",
    }
    assert ids == [3, 3, 7]
    assert words == ["dup", "dup"]


def test_preview_never_reports_values():
    secret_stop = "SECRET-STOP-VALUE-42"
    changes = preview(
        stop=secret_stop,
        stop_token_ids=[424242, 424242, 777777],
        bad_words=["SECRET-BAD-WORD", "SECRET-BAD-WORD"],
        seed=-1,
        temperature=1e-3,
    )
    rendered = "\n".join(f"{change.field}:{change.reason}" for change in changes)
    assert secret_stop not in rendered
    assert "424242" not in rendered
    assert "777777" not in rendered
    assert "SECRET-BAD-WORD" not in rendered
    assert {change.field for change in changes} == {
        "temperature",
        "seed",
        "stop",
        "stop_token_ids",
        "bad_words",
    }


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": -0.1},
        {"temperature": 2.000001},
        {"temperature": float("inf")},
        {"temperature": float("nan")},
        {"top_p": 0.0},
        {"n": 0},
        {"max_tokens": 0},
        {"stop": [""]},
        {"stop_token_ids": [1, "x"]},
        {"bad_words": [""]},
        {"thinking_token_budget": 1.5},
        {"thinking_token_budget": True},
        {"top_k": -2},
        {"min_p": 1.5},
        {"logprobs": -2},
        {"repetition_penalty": 0.0},
        {"no_such_field": 1},
    ],
)
def test_preview_invalid_args_raise_existing_errors(kwargs):
    with pytest.raises(Exception) as direct:
        SamplingParams(**kwargs)
    with pytest.raises(type(direct.value)):
        preview(**kwargs)
