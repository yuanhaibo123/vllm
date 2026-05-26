# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen3Next model."""

import os
from collections.abc import Iterable
from itertools import islice

# Layer-3 MHA capture hook
_L3_CAPTURE_FLAG = '/home/ice/llm/tmp/.capture_l3'
_L3_CAPTURE_PATH = '/home/ice/llm/tmp/vllm_l3_dec.pt'
_l3_cap_live: dict | None = None
# Layer-3 MHA decode capture hook
# Phase 1 (prefill): saves k_rope/v_proj. Phase 2 (decode): saves q/k/v/attn_out.
_L3_DEC_CAPTURE_FLAG = '/home/ice/llm/tmp/.capture_l3_dec'
_L3_DEC_CAPTURE_PATH = '/home/ice/llm/tmp/vllm_l3_decode.pt'
_l3_dec_cap: dict | None = None   # shared across phases
# End-to-end capture: PP0 saves embed_out+input_ids; last rank saves pre-norm hs+residual
_E2E_FLAG      = '/home/ice/llm/tmp/.capture_e2e'
_E2E_PP0_PATH  = '/home/ice/llm/tmp/vllm_e2e_pp0.pt'
_E2E_LAST_PATH = '/home/ice/llm/tmp/vllm_e2e_last.pt'

import torch
from torch import nn

from vllm._aiter_ops import rocm_aiter_ops
from vllm.compilation.decorators import support_torch_compile
from vllm.config import (
    CacheConfig,
    ModelConfig,
    VllmConfig,
    get_current_vllm_config,
)
from vllm.distributed import (
    get_ep_group,
    get_pp_group,
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.fused_moe import (
    FusedMoE,
    fused_moe_make_expert_params_mapping,
)
from vllm.model_executor.layers.layernorm import (
    GemmaRMSNorm as Qwen3NextRMSNorm,
)
from vllm.model_executor.layers.linear import (
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.mamba.gdn_linear_attn import GatedDeltaNetAttention
from vllm.model_executor.layers.mamba.mamba_utils import (
    MambaStateCopyFunc,
    MambaStateCopyFuncCalculator,
    MambaStateDtypeCalculator,
    MambaStateShapeCalculator,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.qwen2_moe import Qwen2MoeMLP as Qwen3NextMLP
from vllm.model_executor.models.utils import sequence_parallel_chunk
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.qwen3_next import Qwen3NextConfig

from .interfaces import (
    EagleModelMixin,
    HasInnerState,
    IsHybrid,
    MixtureOfExperts,
    SupportsLoRA,
    SupportsPP,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)

KVCache = tuple[torch.Tensor, torch.Tensor]


class Qwen3NextSparseMoeBlock(nn.Module):
    def __init__(self, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config
        quant_config = vllm_config.quant_config

        self.tp_size = get_tensor_model_parallel_world_size()

        self.ep_group = get_ep_group().device_group
        self.ep_rank = get_ep_group().rank_in_group
        self.ep_size = self.ep_group.size()
        self.n_routed_experts = config.num_experts

        self.is_sequence_parallel = parallel_config.use_sequence_parallel_moe

        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )

        # Load balancing settings.
        vllm_config = get_current_vllm_config()
        eplb_config = vllm_config.parallel_config.eplb_config
        self.enable_eplb = parallel_config.enable_eplb

        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = eplb_config.num_redundant_experts
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size

        self.physical_expert_start = self.ep_rank * self.n_local_physical_experts
        self.physical_expert_end = (
            self.physical_expert_start + self.n_local_physical_experts
        )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.gate",
        )

        self.shared_expert_gate = ReplicatedLinear(
            config.hidden_size,
            1,
            bias=False,
            quant_config=None,
            prefix=f"{prefix}.shared_expert_gate",
        )

        if (
            rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
            or config.shared_expert_intermediate_size <= 0
        ):
            self.shared_expert = None
        else:
            self.shared_expert = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                expert_gate=self.shared_expert_gate,
                is_sequence_parallel=self.is_sequence_parallel,
                prefix=f"{prefix}.shared_expert",
            )

        self.experts = FusedMoE(
            shared_experts=self.shared_expert,
            gate=self.gate,
            num_experts=self.n_routed_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            renormalize=getattr(config, "norm_topk_prob", True),
            quant_config=quant_config,
            prefix=f"{prefix}.experts",
            enable_eplb=self.enable_eplb,
            num_redundant_experts=self.n_redundant_experts,
            is_sequence_parallel=self.is_sequence_parallel,
            n_shared_experts=1 if self.shared_expert is None else None,
            shared_expert_gate=self.shared_expert_gate
            if self.shared_expert is None
            else None,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # NOTE: hidden_states can have either 1D or 2D shape.
        orig_shape = hidden_states.shape
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if self.is_sequence_parallel:
            hidden_states = sequence_parallel_chunk(hidden_states)

        if self.experts.is_internal_router:
            # In this case, the gate/router runs inside the FusedMoE class
            final_hidden_states = self.experts(
                hidden_states=hidden_states, router_logits=hidden_states
            )
        else:
            # router_logits: (num_tokens, n_experts)
            router_logits, _ = self.gate(hidden_states)
            final_hidden_states = self.experts(
                hidden_states=hidden_states, router_logits=router_logits
            )

        if self.is_sequence_parallel:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, 0
            )
            final_hidden_states = final_hidden_states[:num_tokens]

        return final_hidden_states.view(orig_shape)


class Qwen3NextAttention(nn.Module):
    def __init__(
        self,
        config: Qwen3NextConfig,
        model_config: ModelConfig | None = None,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = config.head_dim or (self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        self.attn_output_gate = getattr(config, "attn_output_gate", True)
        import logging as _lg
        _lg.getLogger(__name__).warning(
            "[ATTN_HD] hidden=%d num_heads=%d num_kv=%d head_dim=%d q_size=%d kv_size=%d attn_gate=%s",
            self.hidden_size, self.total_num_heads, self.total_num_kv_heads,
            self.head_dim, self.q_size, self.kv_size, self.attn_output_gate)

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=getattr(config, "qkv_bias", False),
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
            dual_chunk_attention_config=self.dual_chunk_attention_config,
        )

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": self.dual_chunk_attention_config,
            }
            if self.dual_chunk_attention_config
            else {},
        )

        self.q_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3NextRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.layer_idx = extract_layer_index(prefix)

    def forward(
        self,
        positions: torch.Tensor,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
    ):
        global _l3_cap_live, _l3_dec_cap
        _cap = (_l3_cap_live
                if (not torch.compiler.is_compiling()
                    and self.layer_idx == 3
                    and _l3_cap_live is not None)
                else None)
        if _cap is not None:
            _cap['attn_hs_in'] = hidden_states.detach().cpu().float().clone()

        # Decode capture: arm on first call to layer 3 when flag exists
        _is_prefill = hidden_states.shape[0] > 1
        if (not torch.compiler.is_compiling()
                and self.layer_idx == 3
                and os.path.exists(_L3_DEC_CAPTURE_FLAG)):
            if _l3_dec_cap is None:
                _l3_dec_cap = {}
            _dcap = _l3_dec_cap
        else:
            _dcap = None

        qkv, _ = self.qkv_proj(hidden_states)

        if self.attn_output_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )
            orig_shape = q_gate.shape[:-1]
            q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(*orig_shape, -1)
            gate = gate.reshape(*orig_shape, -1)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        if _cap is not None:
            _cap['q_proj'] = q.detach().cpu().float().clone()
            _cap['k_proj'] = k.detach().cpu().float().clone()
            _cap['v_proj'] = v.detach().cpu().float().clone()
            if self.attn_output_gate:
                _cap['gate_proj'] = gate.detach().cpu().float().clone()

        q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(
            -1, self.num_heads * self.head_dim
        )
        k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(
            -1, self.num_kv_heads * self.head_dim
        )

        if _cap is not None:
            _cap['q_normed'] = q.detach().cpu().float().clone()
            _cap['k_normed'] = k.detach().cpu().float().clone()
            _cap['positions'] = positions.detach().cpu().clone()

        q, k = self.rotary_emb(positions, q, k)

        if _cap is not None:
            _cap['q_rope'] = q.detach().cpu().float().clone()
            _cap['k_rope'] = k.detach().cpu().float().clone()

        # Decode capture: save per-phase data
        if _dcap is not None:
            if _is_prefill:
                # Phase 1: save full prefill KV so we can reconstruct attention at decode
                _dcap['prefill_k_rope'] = k.detach().cpu().float().clone()  # [T, kv_heads*head_dim]
                _dcap['prefill_v'] = v.detach().cpu().float().clone()       # [T, kv_heads*head_dim]
                _dcap['prefill_positions'] = positions.detach().cpu().clone()
            else:
                # Phase 2 (decode, T=1)
                _dcap['dec_q_rope'] = q.detach().cpu().float().clone()  # [1, heads*head_dim]
                _dcap['dec_k_rope'] = k.detach().cpu().float().clone()  # [1, kv_heads*head_dim]
                _dcap['dec_v'] = v.detach().cpu().float().clone()       # [1, kv_heads*head_dim]
                _dcap['dec_position'] = positions.detach().cpu().clone()

        attn_output = self.attn(q, k, v)

        if _cap is not None:
            _cap['attn_out'] = attn_output.detach().cpu().float().clone()

        # Decode capture: save decode attn_output and finalize
        if _dcap is not None and not _is_prefill:
            _dcap['dec_attn_out'] = attn_output.detach().cpu().float().clone()
            torch.save(_dcap, _L3_DEC_CAPTURE_PATH)
            _l3_dec_cap = None
            try:
                os.unlink(_L3_DEC_CAPTURE_FLAG)
            except OSError:
                pass
            logger.warning('L3 decode capture saved to %s (%d keys)', _L3_DEC_CAPTURE_PATH, len(_dcap))

        if self.attn_output_gate:
            gate = torch.sigmoid(gate)
            attn_output = attn_output * gate

        if _cap is not None:
            _cap['gated_attn_out'] = attn_output.detach().cpu().float().clone()

        output[:], _ = self.o_proj(attn_output)


class Qwen3NextDecoderLayer(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        layer_type: str,
        prefix: str = "",
    ) -> None:
        super().__init__()

        config = vllm_config.model_config.hf_config
        model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        self.layer_type = layer_type
        self.layer_idx = extract_layer_index(prefix)

        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNetAttention(
                config,
                vllm_config=vllm_config,
                prefix=f"{prefix}.linear_attn",
                gqa_interleaved_layout=True,
            )
        elif self.layer_type == "full_attention":
            self.self_attn = Qwen3NextAttention(
                config,
                model_config=model_config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=f"{prefix}.self_attn",
            )
        else:
            raise ValueError(f"Invalid layer_type {self.layer_type}")

        mlp_only_layers = (
            [] if not hasattr(config, "mlp_only_layers") else config.mlp_only_layers
        )
        if (self.layer_idx not in mlp_only_layers) and (
            config.num_experts > 0
            and (self.layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3NextSparseMoeBlock(
                vllm_config=vllm_config,
                prefix=f"{prefix}.mlp",
            )
        else:
            self.mlp = Qwen3NextMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=f"{prefix}.mlp",
            )

        self.input_layernorm = Qwen3NextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3NextRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.layer_scale = getattr(config, "layer_scale", False)
        if self.layer_scale:
            self.attn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                ),
            )
            self.ffn_layer_scale = torch.nn.Parameter(
                torch.zeros(
                    1,
                    1,
                    config.hidden_size,
                ),
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        positions: torch.Tensor = None,
        **kwargs: object,
    ):
        _L0_DEC_FLAG = '/home/ice/llm/tmp/.capture_l0'
        _L0_DEC_PATH = '/home/ice/llm/tmp/vllm_l0_decoder.pt'
        import os as _os
        _dec_capturing = (not torch.compiler.is_compiling()
                          and self.layer_idx == 0
                          and _os.path.exists(_L0_DEC_FLAG))
        if _dec_capturing:
            _dcap = {'dec_in': hidden_states.detach().cpu().float().clone(),
                     'dec_residual_in': residual.detach().cpu().float().clone() if residual is not None else None}

        # Layer-3 MHA capture
        global _l3_cap_live
        _l3_capturing = (not torch.compiler.is_compiling()
                         and self.layer_idx == 3
                         and _os.path.exists(_L3_CAPTURE_FLAG))
        if _l3_capturing:
            _l3_cap_live = {
                'dec_in':         hidden_states.detach().cpu().float().clone(),
                'dec_residual_in': residual.detach().cpu().float().clone() if residual is not None else None,
            }

        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        if _dec_capturing:
            _dcap['after_input_ln'] = hidden_states.detach().cpu().float().clone()
        if _l3_capturing:
            _l3_cap_live['after_input_ln'] = hidden_states.detach().cpu().float().clone()

        self_attention_output = torch.empty_like(hidden_states)
        if self.layer_type == "linear_attention":
            self.linear_attn(
                hidden_states=hidden_states,
                output=self_attention_output,
            )
        elif self.layer_type == "full_attention":
            self.self_attn(
                hidden_states=hidden_states,
                output=self_attention_output,
                positions=positions,
            )
        else:
            raise ValueError("Invalid layer_type")
        hidden_states = self_attention_output

        if _dec_capturing:
            _dcap['after_attn'] = hidden_states.detach().cpu().float().clone()
        if _l3_capturing:
            _l3_cap_live['after_attn'] = hidden_states.detach().cpu().float().clone()

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                hidden_states = hidden_states * (
                    self.attn_layer_scale.to(hidden_states.dtype) + 1
                )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)

        if _dec_capturing:
            _dcap['after_post_attn_ln'] = hidden_states.detach().cpu().float().clone()
            _dcap['residual_after_post_ln'] = residual.detach().cpu().float().clone()
        if _l3_capturing:
            _l3_cap_live['after_post_attn_ln'] = hidden_states.detach().cpu().float().clone()
            _l3_cap_live['residual_after_post_ln'] = residual.detach().cpu().float().clone()

        hidden_states = self.mlp(hidden_states)

        if _dec_capturing:
            _dcap['after_mlp'] = hidden_states.detach().cpu().float().clone()
            torch.save(_dcap, _L0_DEC_PATH)
        if _l3_capturing:
            _l3_cap_live['after_mlp'] = hidden_states.detach().cpu().float().clone()
            torch.save(_l3_cap_live, _L3_CAPTURE_PATH)
            _l3_cap_live = None
            try:
                _os.unlink(_L3_CAPTURE_FLAG)
            except OSError:
                pass
            import logging as _lg
            _lg.getLogger(__name__).warning(
                'L3 MHA capture saved to %s (%d keys)', _L3_CAPTURE_PATH, len(torch.load(_L3_CAPTURE_PATH, weights_only=True)))

        if self.layer_scale:
            if len(hidden_states.shape) == 2:
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype)[0] + 1
                )
            else:
                assert len(hidden_states.shape) == len(self.ffn_layer_scale.shape), (
                    f"shape must be the same {len(hidden_states.shape)}, "
                    f"{len(self.ffn_layer_scale.shape)}"
                )
                hidden_states = hidden_states * (
                    self.ffn_layer_scale.to(hidden_states.dtype) + 1
                )

        return hidden_states, residual


@support_torch_compile
class Qwen3NextModel(nn.Module, EagleModelMixin):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        config: Qwen3NextConfig = vllm_config.model_config.hf_text_config
        parallel_config = vllm_config.parallel_config

        eplb_config = parallel_config.eplb_config
        self.num_redundant_experts = eplb_config.num_redundant_experts

        self.config = config

        self.vocab_size = config.vocab_size

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        def get_layer(prefix: str):
            return Qwen3NextDecoderLayer(
                vllm_config,
                layer_type=config.layer_types[extract_layer_index(prefix)],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers, get_layer, prefix=f"{prefix}.layers"
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

        if get_pp_group().is_last_rank:
            self.norm = Qwen3NextRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
        _layer_dbg = int(os.environ.get("VLLM_LAYER_DBG", "0"))
        # Layer boundary capture.
        # TRIGGER: touch /home/ice/llm/tmp/.capture_layer
        # LAYERS:  env VLLM_CAPTURE_LAYERS=N1,N2,...  OR  file .capture_layers_list (one line)
        # OUTPUT:  /home/ice/llm/tmp/vllm_layer_N.pt per layer
        # NOTE:    requires --enforce-eager (CUDA graph replay skips Python forward)
        _CAP_FLAG = '/home/ice/llm/tmp/.capture_layer'
        _cap_env = os.environ.get("VLLM_CAPTURE_LAYERS",
                                   os.environ.get("VLLM_CAPTURE_LAYER", "-1"))
        # Also accept layers from file (works even when env var doesn't reach workers)
        _cap_list_file = '/home/ice/llm/tmp/.capture_layers_list'
        if os.path.exists(_cap_list_file):
            try:
                with open(_cap_list_file) as _f:
                    _cap_env = _f.read().strip() or _cap_env
            except OSError:
                pass
        _cap_layers = set(int(x) for x in _cap_env.split(",") if x.strip() not in ("", "-1"))
        _cap_embed_saved = {}
        _cap_layers_remaining = set(_cap_layers)
        if (not torch.compiler.is_compiling()
                and _cap_layers
                and os.path.exists(_CAP_FLAG)
                and get_pp_group().is_first_rank):
            _cap_embed_saved['embed_out'] = hidden_states.detach().cpu()
            _cap_embed_saved['embed_residual'] = residual.detach().cpu() if residual is not None else None
        # E2E capture: PP0 saves input_ids + embed_out
        if (not torch.compiler.is_compiling()
                and get_pp_group().is_first_rank
                and input_ids is not None
                and os.path.exists(_E2E_FLAG)):
            torch.save({
                'input_ids': input_ids.detach().cpu(),
                'positions': positions.detach().cpu(),
                'embed_out': hidden_states.detach().cpu().float(),
            }, _E2E_PP0_PATH)
            logger.warning('[E2E] embed saved  ids=%s', list(input_ids.shape))
        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
            if _layer_dbg:
                r = residual if residual is not None else hidden_states
                logger.warning("[LAYER_DBG] layer=%d resid_norm=%.4f hs_norm=%.4f",
                               layer_idx, r.float().norm(dim=-1).mean().item(),
                               hidden_states.float().norm(dim=-1).mean().item())
            if (not torch.compiler.is_compiling()
                    and layer_idx in _cap_layers_remaining
                    and os.path.exists(_CAP_FLAG)):
                _out = '/home/ice/llm/tmp/vllm_layer_%d.pt' % layer_idx
                torch.save({
                    'hidden_states': hidden_states.detach().cpu(),
                    'residual': residual.detach().cpu() if residual is not None else None,
                    'layer_idx': layer_idx,
                    **_cap_embed_saved,
                }, _out)
                _cap_layers_remaining.discard(layer_idx)
                logger.warning('[CAPTURE] layer %d boundary saved to %s  hs=%s',
                               layer_idx, _out, list(hidden_states.shape))

            # ── Decode-step capture ──────────────────────────────────────────
            # TRIGGER: echo "N1,N2,..." > /home/ice/llm/tmp/.capture_decode_layers
            #          then touch /home/ice/llm/tmp/.capture_decode
            # OUTPUT:  /home/ice/llm/tmp/vllm_dec_s{step}_l{layer}.pt
            # Saves up to _DEC_CAP_MAX_STEPS decode steps (T=1 forwards only).
            # Steps are counted per-process via a small counter file.
            _DEC_FLAG  = '/home/ice/llm/tmp/.capture_decode'
            _DEC_STEPS = '/home/ice/llm/tmp/.capture_decode_step'   # counter
            _DEC_MAX   = 30
            if (not torch.compiler.is_compiling()
                    and hidden_states.shape[0] == 1          # decode (T=1)
                    and os.path.exists(_DEC_FLAG)
                    and get_pp_group().is_first_rank):
                # Read/increment step counter
                try:
                    with open(_DEC_STEPS) as _sf:
                        _step = int(_sf.read().strip())
                except Exception:
                    _step = 0
                # Read layer list
                _DEC_LIST = '/home/ice/llm/tmp/.capture_decode_layers'
                try:
                    with open(_DEC_LIST) as _lf:
                        _dec_layers = set(int(x) for x in _lf.read().split(',') if x.strip())
                except Exception:
                    _dec_layers = set()
                if _step < _DEC_MAX and layer_idx in _dec_layers:
                    _out = '/home/ice/llm/tmp/vllm_dec_s%d_l%d.pt' % (_step, layer_idx)
                    torch.save({
                        'hidden_states': hidden_states.detach().cpu(),
                        'residual': residual.detach().cpu() if residual is not None else None,
                        'layer_idx': layer_idx,
                        'step': _step,
                        'position': positions.detach().cpu(),
                    }, _out)
                # Increment counter after last watched layer of this step
                if _dec_layers and layer_idx == max(_dec_layers):
                    _step += 1
                    with open(_DEC_STEPS, 'w') as _sf:
                        _sf.write(str(_step))
                    if _step >= _DEC_MAX:
                        try:
                            os.remove(_DEC_FLAG)
                        except OSError:
                            pass
                        logger.warning('[DEC_CAPTURE] done — %d steps captured', _step)
            self._maybe_add_hidden_state(
                aux_hidden_states, layer_idx + 1, hidden_states, residual
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        # Last PP rank: remove capture flag now that all PP workers have run
        if (not torch.compiler.is_compiling()
                and _cap_layers
                and os.path.exists(_CAP_FLAG)):
            os.remove(_CAP_FLAG)
            logger.warning('[CAPTURE] flag removed by last PP rank after all layers')
        # E2E capture: last rank saves pre-final-norm hidden_states + residual
        if (not torch.compiler.is_compiling() and os.path.exists(_E2E_FLAG)):
            torch.save({
                'hidden_states': hidden_states.detach().cpu().float(),
                'residual': residual.detach().cpu().float() if residual is not None else None,
            }, _E2E_LAST_PATH)
            try:
                os.remove(_E2E_FLAG)
            except OSError:
                pass
            logger.warning('[E2E] pre-norm saved  hs=%s', list(hidden_states.shape))
        hidden_states, _ = self.norm(hidden_states, residual)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        num_experts = getattr(self.config, "num_experts", 0)
        if rocm_aiter_ops.is_fusion_moe_shared_experts_enabled():
            num_experts += 1
        return fused_moe_make_expert_params_mapping(
            self,
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=num_experts,
            num_redundant_experts=self.num_redundant_experts,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        expert_params_mapping = self.get_expert_mapping()

        is_fse = rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
        num_routed = getattr(self.config, "num_experts", 0)

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue

            if name.startswith("mtp."):
                continue

            # Remapping the name of FP8 kv-scale.
            if name.endswith("scale"):
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue

            # FSE: remap shared_expert weights to the fused expert slot
            if is_fse and "mlp.shared_expert." in name:
                name = name.replace(
                    "mlp.shared_expert.",
                    f"mlp.experts.{num_routed}.",
                )

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                if "mlp.experts" in name:
                    continue

                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip layers on other devices.
                if is_pp_missing_parameter(name, self):
                    continue
                # name = apply_attn_prefix(name, params_dict)
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    # Skip layers on other devices.
                    if is_pp_missing_parameter(name, self):
                        continue
                    # Skip loading extra bias for GPTQ models.
                    if (
                        name.endswith(".bias") or name.endswith("_bias")
                    ) and name not in params_dict:
                        continue
                    if name not in params_dict:
                        continue
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if is_pp_missing_parameter(name, self):
                        continue
                    if name not in params_dict:
                        logger.warning_once(
                            f"Parameter {name} not found in params_dict, skip loading"
                        )
                        continue
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class QwenNextMixtureOfExperts(MixtureOfExperts):
    def update_physical_experts_metadata(
        self,
        num_physical_experts: int,
        num_local_physical_experts: int,
    ) -> None:
        assert self.num_local_physical_experts == num_local_physical_experts
        self.num_physical_experts = num_physical_experts
        self.num_local_physical_experts = num_local_physical_experts
        self.num_redundant_experts = num_physical_experts - self.num_logical_experts
        for layer in self.model.layers:
            if isinstance(layer.mlp, Qwen3NextSparseMoeBlock):
                moe = layer.mlp
                moe.n_local_physical_experts = num_local_physical_experts
                moe.n_physical_experts = num_physical_experts
                moe.n_redundant_experts = self.num_redundant_experts
                moe.experts.update_expert_map()

    def set_moe_parameters(self):
        self.expert_weights = []

        self.moe_layers = []
        example_moe = None
        for layer in self.model.layers:
            if isinstance(layer, Qwen3NextDecoderLayer) and isinstance(
                layer.mlp, Qwen3NextSparseMoeBlock
            ):
                example_moe = layer.mlp
                self.moe_layers.append(layer.mlp.experts)

        if example_moe is None:
            raise RuntimeError("No Qwen3Next layer found in the model.layers.")

        # Set MoE hyperparameters
        self.num_moe_layers = len(self.moe_layers)
        self.num_expert_groups = 1
        self.num_shared_experts = 0
        self.num_logical_experts = example_moe.n_logical_experts
        self.num_physical_experts = example_moe.n_physical_experts
        self.num_local_physical_experts = example_moe.n_local_physical_experts
        self.num_routed_experts = example_moe.n_routed_experts
        self.num_redundant_experts = example_moe.n_redundant_experts


class Qwen3NextForCausalLM(
    nn.Module,
    HasInnerState,
    SupportsLoRA,
    SupportsPP,
    QwenNextMixtureOfExperts,
    IsHybrid,
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkvz"],
        "in_proj_ba": ["in_proj_ba"],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        cache_config = vllm_config.cache_config

        scheduler_config = vllm_config.scheduler_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen3Next currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )
        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.scheduler_config = scheduler_config
        self.model = Qwen3NextModel(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
        )

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        # Set MoE hyperparameters
        self.set_moe_parameters()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: object,
    ):
        hidden_states = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

        return hidden_states

    @classmethod
    def get_mamba_state_dtype_from_config(
        cls,
        vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype,
            vllm_config.cache_config.mamba_cache_dtype,
            vllm_config.cache_config.mamba_ssm_cache_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config
            else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple[MambaStateCopyFunc, MambaStateCopyFunc]:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp."],
        )
        return loader.load_weights(weights)

    def get_expert_mapping(self) -> list[tuple[str, str, int, str]]:
        return self.model.get_expert_mapping()
