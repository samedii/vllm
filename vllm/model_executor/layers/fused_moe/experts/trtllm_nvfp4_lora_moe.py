# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
LoRA-aware FlashInfer TRT-LLM-Gen NVFP4 MoE experts (experimental fork branch).

Keeps the base expert GEMMs on ``trtllm_fp4_block_scale_routed_moe`` when
``--enable-lora`` is on, instead of falling back to Marlin W4A16.

  * W13 (gate_up) LoRA: computed out of kernel by punica into a bf16
    ``[T, top_k, 2I]`` delta and handed to the routed kernel as
    ``gemm1_lora_delta`` (FlashInfer PR #3987), which adds it before SwiGLU.
  * W2 (down) LoRA: depends on the mode selected by
    ``VLLM_NVFP4_MOE_LORA_TRTLLM``:
      - ``w13_only``: the W2 delta is DROPPED (bounds the fused-path cost,
        does not match a W2-trained adapter). The kernel finalizes in place.
      - ``full``: requires ``VLLM_NVFP4_MOE_PER_TOKEN_ACT=1``. In per-token
        activation mode the kernel returns the post-SwiGLU FC1 activation as
        bf16 with ``do_finalize=False``; the shared base ``apply`` unpermutes
        it, runs the punica W2 delta and fuses the finalize.
        In the static activation-scale mode the FC1 activation is returned
        as FP4 codes without its block scales, so ``full`` is refused.

Selection lives in ``oracle/nvfp4.py`` behind the same env flag; nothing
changes when the flag is unset.
"""

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import envs
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEQuantConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.experts.trtllm_lora_moe import (
    _TrtLlmLoRAExpertsBase,
)
from vllm.model_executor.layers.fused_moe.experts.trtllm_nvfp4_moe import (
    TrtLlmNvFp4ExpertsBase,
)
from vllm.model_executor.layers.fused_moe.utils import fi_moe_largest_bucket
from vllm.model_executor.layers.quantization.utils.flashinfer_utils import (
    activation_to_flashinfer_int,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import QuantKey

logger = init_logger(__name__)

LORA_TRTLLM_MODES = ("w13_only", "full")


def nvfp4_lora_trtllm_mode() -> str | None:
    """Return the requested mode or None when the fork path is off."""
    mode = envs.VLLM_NVFP4_MOE_LORA_TRTLLM
    if not mode or mode == "off":
        return None
    if mode not in LORA_TRTLLM_MODES:
        raise ValueError(
            f"VLLM_NVFP4_MOE_LORA_TRTLLM={mode!r}; expected one of "
            f"{LORA_TRTLLM_MODES} or 'off'."
        )
    return mode


class TrtLlmNvFp4LoRAExperts(TrtLlmNvFp4ExpertsBase, _TrtLlmLoRAExpertsBase):
    """NVFP4 (ModelOpt static weights, NVFP4 activations) TRT-LLM MoE + LoRA."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        per_token_activation: bool = False,
    ):
        # Both bases are plain classes that do not chain to each other:
        # run the LoRA/modular base first (registers the modular kernel
        # state), then the NVFP4 base (scales, SwiGLU constants).
        _TrtLlmLoRAExpertsBase.__init__(self, moe_config, quant_config)
        TrtLlmNvFp4ExpertsBase.__init__(
            self, moe_config, quant_config, per_token_activation
        )
        mode = nvfp4_lora_trtllm_mode()
        assert mode is not None, "TrtLlmNvFp4LoRAExperts built with the flag off"
        self.lora_mode = mode
        if mode == "full" and not per_token_activation:
            raise ValueError(
                "VLLM_NVFP4_MOE_LORA_TRTLLM=full needs the bf16 FC1 activation, "
                "which the routed NVFP4 kernel only returns in per-token "
                "activation mode: also set VLLM_NVFP4_MOE_PER_TOKEN_ACT=1."
            )
        if mode == "w13_only":
            logger.warning_once(
                "TrtLlmNvFp4LoRAExperts mode=w13_only: the W2 (down_proj) "
                "LoRA delta is DROPPED for routed experts. Only for cost "
                "measurement; outputs differ from the trained adapter."
            )
        logger.info_once(
            "TrtLlmNvFp4LoRAExperts enabled: mode=%s per_token_activation=%s",
            mode,
            per_token_activation,
        )

    # ---- capability gates (resolve the diamond explicitly) ----

    @staticmethod
    def _supports_current_device() -> bool:
        return TrtLlmNvFp4ExpertsBase._supports_current_device()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return False  # the LoRA finalize assumes a gated MLP

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None, activation_key: QuantKey | None
    ) -> bool:
        return TrtLlmNvFp4ExpertsBase._supports_quant_scheme(
            weight_key, activation_key
        )

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation == MoEActivation.SILU

    @staticmethod
    def _supports_routing_method(routing_method, weight_key, activation_key) -> bool:
        return routing_method in [
            RoutingMethodType.DeepSeekV3,
            RoutingMethodType.Llama4,
            RoutingMethodType.Renormalize,
            RoutingMethodType.RenormalizeNaive,
        ]

    @staticmethod
    def _supports_shape(hidden_dim: int) -> bool:
        return hidden_dim % 256 == 0

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @property
    def expects_unquantized_inputs(self) -> bool:
        # Static mode: prepare() quantizes with the calibrated global scale
        # and apply() receives packed FP4 + block scales. Per-token mode:
        # apply() receives bf16 and quantizes itself.
        return self.per_token_activation

    # workspace_shapes / moe_problem_size / finalize_weight_and_reduce_impl /
    # _supports_parallel_config / _supports_router_logits_dtype come from
    # _TrtLlmLoRAExpertsBase (MRO: NvFp4 base defines none of them).

    # ---- kernel call ----

    def invoke_routed_moe(
        self,
        *,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids_and_weights: tuple[torch.Tensor, torch.Tensor],
        gemm1_lora_delta: torch.Tensor | None,
        global_num_experts: int,
        a1q_scale: torch.Tensor | None,
        output: torch.Tensor,
        do_finalize: bool | None = None,
        activation: MoEActivation = MoEActivation.SILU,
    ) -> list[torch.Tensor]:
        import flashinfer

        assert self.quant_config.w1_scale is not None
        assert self.quant_config.w2_scale is not None

        if self.per_token_activation:
            hidden_states, block_scale, per_token_scale = (
                self._quantize_per_token_input(hidden_states)
            )
        else:
            assert a1q_scale is not None, "static NVFP4 needs a1q_scale"
            block_scale, per_token_scale = a1q_scale, None

        if do_finalize is None:
            do_finalize = gemm1_lora_delta is None

        ret = flashinfer.fused_moe.trtllm_fp4_block_scale_routed_moe(
            topk_ids=topk_ids_and_weights,
            routing_bias=None,
            hidden_states=hidden_states,
            hidden_states_scale=block_scale.view(torch.float8_e4m3fn).reshape(
                *hidden_states.shape[:-1], -1
            ),
            gemm1_weights=w1,
            gemm1_weights_scale=self.quant_config.w1_scale.view(torch.float8_e4m3fn),
            gemm1_bias=None,
            gemm1_alpha=self.gemm1_alpha,
            gemm1_beta=self.gemm1_beta,
            gemm1_clamp_limit=self.gemm1_clamp_limit,
            gemm2_weights=w2,
            gemm2_weights_scale=self.quant_config.w2_scale.view(torch.float8_e4m3fn),
            gemm2_bias=None,
            output1_scale_scalar=self.g1_scale_c,
            output1_scale_gate_scalar=self.quant_config.g1_alphas,
            output2_scale_scalar=self.quant_config.g2_alphas,
            num_experts=global_num_experts,
            top_k=self.topk,
            n_group=0,
            topk_group=0,
            intermediate_size=self.intermediate_size_per_partition,
            local_expert_offset=self.ep_rank * self.local_num_experts,
            local_num_experts=self.local_num_experts,
            routed_scaling_factor=None,
            routing_method_type=1,  # not used: routing is precomputed
            do_finalize=do_finalize,
            activation_type=activation_to_flashinfer_int(activation),
            per_token_scale=per_token_scale,
            output=output if do_finalize else None,
            gemm1_lora_delta=gemm1_lora_delta,
            tune_max_num_tokens=min(
                fi_moe_largest_bucket(self.moe_config), self._get_chunk_size()
            ),
        )
        if do_finalize:
            return [output]
        ret = list(ret)
        if len(ret) != 4:
            raise RuntimeError(
                "trtllm_fp4_block_scale_routed_moe(do_finalize=False, "
                f"gemm1_lora_delta=set) returned {len(ret)} tensors; expected "
                "[gemm2_output, expert_weights, expanded_idx_to_permuted_idx, "
                "gemm1_activation_output]."
            )
        return ret

    def _static_lora_chunk_tokens(self) -> int:
        """Largest token chunk for the static-scale (E2m1 out) kernel with a
        LoRA delta.

        FlashInfer 0.6.18 advertises tile_N 192 for static E2m1 activations
        (``FP4BlockScaleLauncher::getSupportedTileNums``) and, with autotune
        off, picks the smaller neighbour of ``nextPow2(tokens*top_k/E)`` as
        the default tile - 192 as soon as tokens*top_k/E > 128. The cubin set
        has no ``Bmm_E2m1_E2m1E2m1_*_t128x192*_biasBfloat16Mn`` kernel, so
        that default tile fails runner construction ("No kernel found for the
        given options ... mTileSize: 192, mBiasType: 3") on the LoRA path
        only. Chunking so that tokens*top_k/E <= 128 keeps the default tile
        at <= 128, where LoRA-bias kernels exist. Per-token mode never
        advertises 192 and is unaffected. Rounded down to a multiple of 128
        so per-token FP4 block-scale rows stay aligned.
        """
        if self.per_token_activation:
            return self._get_chunk_size()
        max_tokens = (128 * self.moe_config.num_experts) // self.topk
        return max(128, (max_tokens // 128) * 128)

    # ---- forward ----

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        assert activation == MoEActivation.SILU
        assert not apply_router_weight_on_input
        # DeepEP produces int64 indexes; the routed kernel wants int32.
        topk_ids = topk_ids.to(dtype=torch.int32)
        num_tokens = hidden_states.size(0)
        assert num_tokens <= self._get_chunk_size(), (
            "LoRA NVFP4 path does not chunk; raise chunking if this fires"
        )
        lora_context = self._lora_context

        # No LoRA wrapper on this layer (e.g. the MTP drafter's MoE block,
        # which shares the experts class but is never wrapped) or no LoRA
        # rows in this batch: plain base kernel, finalized in place.
        if lora_context is None or self._batch_has_no_lora(lora_context):
            self.invoke_routed_moe(
                hidden_states=hidden_states,
                w1=w1,
                w2=w2,
                topk_ids_and_weights=(topk_ids, topk_weights),
                gemm1_lora_delta=None,
                global_num_experts=global_num_experts,
                a1q_scale=a1q_scale,
                output=output,
                activation=activation,
            )
            return

        if self.lora_mode == "full":
            # W13 fused + W2 out of kernel + fused finalize (bf16 FC1 output
            # is available because per_token_activation is on).
            return _TrtLlmLoRAExpertsBase.apply(
                self,
                output,
                hidden_states,
                w1,
                w2,
                topk_weights,
                topk_ids,
                activation,
                global_num_experts,
                expert_map,
                a1q_scale,
                a2_scale,
                workspace13,
                workspace2,
                expert_tokens_meta,
                apply_router_weight_on_input,
            )

        # ---- w13_only: fused W13 delta, kernel finalizes, W2 delta dropped ----
        top_k = self.topk
        intermediate_size = self.intermediate_size_per_partition
        K = output.size(1)
        w1_cfg = torch.empty(
            (self.local_num_experts, 2 * intermediate_size, K),
            device="meta",
            dtype=torch.bfloat16,
        )
        w2_cfg = torch.empty(
            (self.local_num_experts, K, intermediate_size),
            device="meta",
            dtype=torch.bfloat16,
        )
        gemm1_lora_delta = torch.zeros(
            num_tokens,
            top_k,
            2 * intermediate_size,
            dtype=torch.bfloat16,
            device=hidden_states.device,
        )
        lora_x = hidden_states
        if not self.expects_unquantized_inputs:
            orig = lora_context.original_hidden_states
            assert orig is not None and orig.shape[0] == num_tokens, (
                "quantized trtllm LoRA path requires original_hidden_states"
            )
            lora_x = orig
        self.apply_w13_lora(
            lora_context,
            y=gemm1_lora_delta,
            x=lora_x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
            expert_map=expert_map,
            w1=w1_cfg,
            w2=w2_cfg,
            num_tokens=num_tokens,
            top_k_num=top_k,
            add_inputs=False,
            swap_w13_slices=True,
        )
        chunk = self._static_lora_chunk_tokens()
        for start in range(0, num_tokens, chunk):
            end = min(start + chunk, num_tokens)
            self.invoke_routed_moe(
                hidden_states=hidden_states[start:end],
                w1=w1,
                w2=w2,
                topk_ids_and_weights=(
                    topk_ids[start:end],
                    topk_weights[start:end],
                ),
                gemm1_lora_delta=gemm1_lora_delta[start:end],
                global_num_experts=global_num_experts,
                a1q_scale=None if a1q_scale is None else a1q_scale[start:end],
                output=output[start:end],
                do_finalize=True,
                activation=activation,
            )
