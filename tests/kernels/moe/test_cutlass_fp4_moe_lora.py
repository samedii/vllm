# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for CutlassExpertsFp4LoRA: LoRA injection points of the vLLM
CUTLASS NVFP4 MoE kernel.

The punica MoE LoRA kernels are replaced by pure-torch per-expert LoRA math
that follows the same contract (token-order flat pair rows, W2 multiplies the
routed weight itself), so the test exercises the sorted<->token-order
permutation plumbing and the pre-weighted W2 add, not the Triton kernels.
"""

import types

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.utils import make_dummy_moe_config, make_test_weights
from tests.kernels.quantization.nvfp4_utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    dequantize_nvfp4_to_dtype,
)
from tests.kernels.utils import torch_experts, torch_moe
from vllm import _custom_ops as ops
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import nvfp4_moe_quant_config
from vllm.model_executor.layers.fused_moe.experts.cutlass_moe import (
    CutlassExpertsFp4,
    CutlassExpertsFp4LoRA,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.has_device_capability(100):
    pytest.skip(
        "Nvfp4 Requires compute capability of 10 or above.",
        allow_module_level=True,
    )

MNK_FACTORS = [
    (2, 640, 2560),
    (37, 640, 2560),
    (224, 1024, 1024),
]


def _lora_delta(x_rows: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
    # x_rows (p, in), a (out_r, in) -> (p, r), b (out, r)
    return (x_rows.float() @ a.float().t()) @ b.float().t()


def _make_kernel(experts, moe_config, quant_config):
    return mk.FusedMoEKernel(
        maybe_make_prepare_finalize(
            moe=moe_config,
            quant_config=quant_config,
            allow_new_interface=True,
            use_monolithic=False,
        ),
        experts,
    )


@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.parametrize("e", [16, 64])
@pytest.mark.parametrize("topk", [2, 6])
@pytest.mark.parametrize("rank", [8])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("invalid_frac", [0.0, 0.25])
@torch.inference_mode()
def test_cutlass_fp4_moe_lora_hooks(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    rank: int,
    dtype: torch.dtype,
    invalid_frac: float,
    workspace_init,
):
    set_random_seed(7)
    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        quant_blocksize = 16
        a = torch.randn((m, k), device="cuda", dtype=dtype) / 10

        (_, w1_q, w1_blockscale, w1_gs), (_, w2_q, w2_blockscale, w2_gs) = (
            make_test_weights(
                e,
                n,
                k,
                in_dtype=dtype,
                quant_dtype="nvfp4",
                block_shape=None,
                per_out_ch_quant=False,
            )
        )
        score = torch.randn((m, e), device="cuda", dtype=dtype)
        topk_weights, topk_ids, _ = fused_topk(a, score, topk, renormalize=False)
        if invalid_frac > 0:
            # Invalid routes (as vLLM emits for dropped / masked slots): the
            # CUTLASS data kernel gives them an out-of-range c_map sentinel.
            drop = torch.rand(topk_ids.shape, device="cuda") < invalid_frac
            drop[0, 0] = True
            topk_ids = topk_ids.masked_fill(drop, -1)

        a1_gs = torch.ones((e,), device="cuda", dtype=torch.float32)
        a2_gs = torch.ones((e,), device="cuda", dtype=torch.float32)
        quant_config = nvfp4_moe_quant_config(
            g1_alphas=(1 / w1_gs),
            g2_alphas=(1 / w2_gs),
            a1_gscale=a1_gs,
            a2_gscale=a2_gs,
            w1_scale=w1_blockscale,
            w2_scale=w2_blockscale,
        )
        moe_config = make_dummy_moe_config()

        # LoRA weights, per expert. Scaled so the adapter delta is well above
        # the NVFP4 quantization noise of the base path.
        lora_scale = 0.5
        a13 = torch.randn((e, rank, k), device="cuda", dtype=dtype) / k**0.5
        b13 = torch.randn((e, 2 * n, rank), device="cuda", dtype=dtype) * lora_scale
        a2 = torch.randn((e, rank, n), device="cuda", dtype=dtype) / n**0.5
        b2 = torch.randn((e, k, rank), device="cuda", dtype=dtype) * lora_scale

        flat_ids = topk_ids.reshape(-1).long()  # flat pair -> expert
        pair_tok = torch.arange(m * topk, device="cuda") // topk

        experts = CutlassExpertsFp4LoRA(moe_config=moe_config, quant_config=quant_config)
        experts.set_lora_context(types.SimpleNamespace(original_hidden_states=None))  # type: ignore[arg-type]
        seen: dict = {}

        def fake_w13(ctx, *, y, x, topk_ids, topk_weights, expert_map, w1, w2,
                     num_tokens, top_k_num, add_inputs=True, swap_w13_slices=False):
            assert y.shape == (m * topk, 2 * n) and x.shape == (m, k)
            assert torch.count_nonzero(y) == 0, "W13 delta buffer must be zeroed"
            for ex in range(e):
                mask = flat_ids == ex
                if mask.any():
                    y[mask] += _lora_delta(x[pair_tok[mask]], a13[ex], b13[ex]).to(y.dtype)
            seen["w13"] = True
            return None, None, None, None

        def fake_w2(ctx, *, y, x, topk_weights, sorted_token_ids_lora, expert_ids_lora,
                    num_tokens_post_padded_lora, token_lora_mapping, num_tokens, w1, w2,
                    top_k_num, add_inputs=True):
            assert y.shape == (m, topk, k) and x.shape == (m * topk, n)
            yf = y.view(m * topk, k)
            wf = topk_weights.reshape(-1, 1).float()
            for ex in range(e):
                mask = flat_ids == ex
                if mask.any():
                    yf[mask] += (wf[mask] * _lora_delta(x[mask], a2[ex], b2[ex])).to(y.dtype)
            seen["w2"] = True

        experts.apply_w13_lora = fake_w13  # type: ignore[method-assign]
        experts.apply_w2_lora = fake_w2  # type: ignore[method-assign]

        kernel = _make_kernel(experts, moe_config, quant_config)
        lora_output = kernel.apply(
            hidden_states=a,
            w1=w1_q,
            w2=w2_q,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            global_num_experts=e,
            activation=mk.MoEActivation.SILU,
            apply_router_weight_on_input=False,
            expert_map=None,
        )
        assert seen == {"w13": True, "w2": True}

        base_kernel = _make_kernel(
            CutlassExpertsFp4(moe_config=moe_config, quant_config=quant_config),
            moe_config,
            quant_config,
        )
        base_output = base_kernel.apply(
            hidden_states=a,
            w1=w1_q,
            w2=w2_q,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            global_num_experts=e,
            activation=mk.MoEActivation.SILU,
            apply_router_weight_on_input=False,
            expert_map=None,
        )

        # Reference: torch MoE on dequantized weights with the LoRA merged in.
        a_global_scale = (
            (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / torch.amax(a.flatten(), dim=-1)
        ).to(torch.float32)
        a_fp4, a_scale_interleaved = ops.scaled_fp4_quant(a, a_global_scale)
        a_in_dtype = dequantize_nvfp4_to_dtype(
            a_fp4, a_scale_interleaved, a_global_scale,
            dtype=a.dtype, device=a.device, block_size=quant_blocksize,
        )
        w1_d = torch.empty((e, 2 * n, k), device="cuda", dtype=dtype)
        w2_d = torch.empty((e, k, n), device="cuda", dtype=dtype)
        for idx in range(e):
            w1_d[idx] = dequantize_nvfp4_to_dtype(
                w1_q[idx], w1_blockscale[idx], w1_gs[idx],
                dtype=dtype, device=w1_q.device, block_size=quant_blocksize,
            )
            w2_d[idx] = dequantize_nvfp4_to_dtype(
                w2_q[idx], w2_blockscale[idx], w2_gs[idx],
                dtype=dtype, device=w2_q.device, block_size=quant_blocksize,
            )
        w1_merged = (w1_d.float() + b13.float() @ a13.float()).to(dtype)
        w2_merged = (w2_d.float() + b2.float() @ a2.float()).to(dtype)
        # torch_experts skips ids outside [0, e), like the kernels do.
        ref_lora = torch_experts(a_in_dtype, w1_merged, w2_merged, topk_weights, topk_ids)
        ref_base = torch_experts(a_in_dtype, w1_d, w2_d, topk_weights, topk_ids)

        # The adapter must actually move the output (guards a trivially
        # passing test) ...
        applied = (lora_output - base_output).float().norm()
        expected = (ref_lora - ref_base).float().norm()
        assert expected > 0.05 * ref_base.float().norm()
        assert applied > 0.5 * expected
        # ... and the LoRA output must match the merged reference to the
        # same tolerance the base kernel test uses.
        torch.testing.assert_close(ref_lora, lora_output, atol=1e-1, rtol=1e-1)
        torch.testing.assert_close(ref_base, base_output, atol=1e-1, rtol=1e-1)


@pytest.mark.parametrize("m,n,k", [(37, 640, 2560)])
@torch.inference_mode()
def test_cutlass_fp4_moe_lora_class_without_context_matches_base(
    m, n, k, workspace_init
):
    """No LoRA context (e.g. the MTP drafter's MoE) -> bit-identical to base."""
    e, topk, dtype = 16, 4, torch.bfloat16
    set_random_seed(7)
    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        a = torch.randn((m, k), device="cuda", dtype=dtype) / 10
        (_, w1_q, w1_blockscale, w1_gs), (_, w2_q, w2_blockscale, w2_gs) = (
            make_test_weights(
                e, n, k, in_dtype=dtype, quant_dtype="nvfp4",
                block_shape=None, per_out_ch_quant=False,
            )
        )
        score = torch.randn((m, e), device="cuda", dtype=dtype)
        topk_weights, topk_ids, _ = fused_topk(a, score, topk, renormalize=False)
        quant_config = nvfp4_moe_quant_config(
            g1_alphas=(1 / w1_gs), g2_alphas=(1 / w2_gs),
            a1_gscale=torch.ones((e,), device="cuda", dtype=torch.float32),
            a2_gscale=torch.ones((e,), device="cuda", dtype=torch.float32),
            w1_scale=w1_blockscale, w2_scale=w2_blockscale,
        )
        moe_config = make_dummy_moe_config()
        outs = []
        for cls in (CutlassExpertsFp4, CutlassExpertsFp4LoRA):
            kernel = _make_kernel(
                cls(moe_config=moe_config, quant_config=quant_config),
                moe_config, quant_config,
            )
            outs.append(kernel.apply(
                hidden_states=a, w1=w1_q, w2=w2_q,
                topk_weights=topk_weights, topk_ids=topk_ids,
                global_num_experts=e, activation=mk.MoEActivation.SILU,
                apply_router_weight_on_input=False, expert_map=None,
            ))
        assert torch.equal(outs[0], outs[1])


def test_cutlass_fp4_moe_lora_supports_lora_flag():
    assert not CutlassExpertsFp4.supports_lora()
    assert CutlassExpertsFp4LoRA.supports_lora()
