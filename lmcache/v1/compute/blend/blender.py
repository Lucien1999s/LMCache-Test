# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional, Union

# Third Party
import torch

# First Party
from lmcache import torch_device_type
from lmcache.logging import init_logger
from lmcache.v1.compute.attention.metadata import LMCAttnMetadata
from lmcache.v1.compute.blend.metadata import LMCBlendCommonMetadata, LMCBlendMetadata
from lmcache.v1.compute.models.utils import infer_model_from_vllm
from lmcache.v1.config import LMCacheEngineConfig
# ContextFlow drift/layout/profiling hooks are disabled unless explicitly enabled.
from lmcache.v1.contextflow_drift_analysis import (
    is_enabled as drift_analysis_enabled,
    record_layer_drift,
)
from lmcache.v1.contextflow_kv_layout import record_selected_repair_layer
from lmcache.v1.contextflow_profiler import (
    record_event as cf_record_event,
    span as cf_span,
)

logger = init_logger(__name__)


class LMCBlender:
    """
    Cache-blender backend for LMCache.
    This backend uses the Blender implementation for efficient blending computation.
    """

    def __init__(
        self,
        cache_engine,
        gpu_connector,
        vllm_model,
        config: LMCacheEngineConfig,
    ):
        self.cache_engine = cache_engine
        self.gpu_connector = gpu_connector

        enable_sparse = False
        if config.extra_config is not None:
            enable_sparse = config.extra_config.get("enable_sparse", False)

        self.layerwise_model = infer_model_from_vllm(vllm_model, self, enable_sparse)

        # TODO: remove this hardcode
        self.num_layers = len(vllm_model.model.layers)

        # TODO(Jiayi): support threshold-based blending
        # TODO(Jiayi): support different ratios for different layers
        # TODO(Jiayi): support "skipping blending if hit too short"
        self.common_metadata = LMCBlendCommonMetadata(
            check_layers=config.blend_check_layers,
            recomp_ratios=config.blend_recompute_ratios,
            thresholds=config.blend_thresholds,
        )

        # This will be set during the blending process
        self.metadata = LMCBlendMetadata(
            imp_indices=None,
            attn_mask=None,
            positions=None,
        )

    def process_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        residual: torch.Tensor,
        layer_id: int,
        attn_output: Optional[torch.Tensor],
        attn_metadata: LMCAttnMetadata,
    ):
        logger.debug(f"Blender is processing KV for layer {layer_id}")
        old_k, old_v = self.gpu_connector.get_kv(layer_id)

        if attn_output is None:
            attn_output = torch.empty(
                q.shape,
                dtype=q.dtype,
                device=q.device,
            )

        # perform positional encoding
        if self.metadata.positions is None:
            self.metadata.positions = torch.arange(
                q.shape[0], device=q.device, dtype=torch.int64
            )
        layer = self.layerwise_model.vllm_model.model.layers[layer_id]
        attn_layer = layer.self_attn
        with cf_span(
            "rope_on_new_qk",
            category="blender",
            device="GPU",
            layer_id=layer_id,
            metadata={"tokens": q.shape[0]},
        ):
            q, k = attn_layer.rotary_emb(self.metadata.positions, q, k)

        if layer_id in self.common_metadata.check_layers:
            with cf_span(
                "kv_drift_measurement",
                category="blender",
                device="GPU",
                layer_id=layer_id,
                metadata={"tokens": k.shape[0]},
            ):
                diff_k = torch.sum(
                    (k.to(torch.float32) - old_k.to(torch.float32)) ** 2, dim=[1]
                )
            total_len = diff_k.shape[0]

            assert self.common_metadata.recomp_ratios is not None

            # TODO(Jiayi): remove `[0]` hardcode
            topk_num = int(total_len * self.common_metadata.recomp_ratios[0])
            topk_num = max(topk_num, 1)

            with cf_span(
                "important_token_topk",
                category="blender",
                device="GPU",
                layer_id=layer_id,
                metadata={"tokens": total_len, "topk": topk_num},
            ):
                top_indices = torch.topk(diff_k, k=topk_num).indices
                top_indices, _ = torch.sort(top_indices)

            record_layer_drift(
                request_id=self.metadata.request_id,
                layer_id=layer_id,
                drift_scores=diff_k,
                drift_scope="full_cached_tokens",
                cached_tokens=int(old_k.shape[0]),
                selected_indices=top_indices,
                recompute_ratio=self.common_metadata.recomp_ratios[0],
                scores_are_selected_subset=False,
            )

            with cf_span(
                "important_token_slice",
                category="blender",
                device="GPU",
                layer_id=layer_id,
                metadata={"tokens": total_len, "topk": topk_num},
            ):
                k, v = k[top_indices], v[top_indices]
                q = q[top_indices]
                residual = residual[top_indices]

            logger.debug(f"Number of indices picked: {len(top_indices)}")

            self.metadata.imp_indices = top_indices
            self.metadata.positions = self.metadata.positions[top_indices]
            attn_output = attn_output[:topk_num]

            with cf_span(
                "repair_attention_metadata_update",
                category="blender",
                device="CPU/GPU",
                layer_id=layer_id,
                metadata={"tokens": total_len, "topk": topk_num},
            ):
                attn_metadata.update_from_top_indices(top_indices)

            cf_record_event(
                "important_token_selection_result",
                category="blender",
                device="GPU",
                layer_id=layer_id,
                metadata={
                    "tokens": total_len,
                    "selected_tokens": topk_num,
                    "recompute_ratio": self.common_metadata.recomp_ratios[0],
                },
            )

        if self.metadata.imp_indices is not None:
            if drift_analysis_enabled() and layer_id not in self.common_metadata.check_layers:
                selected_old_k = old_k[self.metadata.imp_indices]
                selected_diff_k = torch.sum(
                    (k.to(torch.float32) - selected_old_k.to(torch.float32)) ** 2,
                    dim=[1],
                )
                record_layer_drift(
                    request_id=self.metadata.request_id,
                    layer_id=layer_id,
                    drift_scores=selected_diff_k,
                    drift_scope="selected_repair_tokens",
                    cached_tokens=int(old_k.shape[0]),
                    selected_indices=self.metadata.imp_indices,
                    recompute_ratio=(
                        self.common_metadata.recomp_ratios[0]
                        if self.common_metadata.recomp_ratios is not None
                        else None
                    ),
                    scores_are_selected_subset=True,
                )
            record_selected_repair_layer(
                request_id=self.metadata.request_id,
                layer_id=layer_id,
                selected_indices=self.metadata.imp_indices,
                slot_mapping=self.metadata.slot_mapping,
                block_size=self.metadata.block_size,
                key_tensor=old_k,
                value_tensor=old_v,
            )
            with cf_span(
                "selective_kv_replacement",
                category="blender",
                device="GPU",
                layer_id=layer_id,
                metadata={"selected_tokens": k.shape[0]},
            ):
                old_k[self.metadata.imp_indices] = k
                old_v[self.metadata.imp_indices] = v
            return q, old_k, old_v, residual, attn_output, attn_metadata
        else:
            return q, k, v, residual, attn_output, attn_metadata

    # NOTE(Jiayi): Exposing this `blend_layer` interface as we might
    # want to ochestrate the blending process elsewhere
    def blend_layer(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Perform layerwiese retrieve + blending.
        """

        # TODO(Jiayi): store is currently not included in this function

        self.metadata.request_id = kwargs.get("req_id")
        self.metadata.slot_mapping = kwargs.get("slot_mapping")
        self.metadata.block_size = kwargs.get("block_size")

        layerwise_model_executor = self.layerwise_model.compute_layer(tokens)
        layerwise_retriever = self.cache_engine.retrieve_layer(tokens, mask, **kwargs)

        with cf_span(
            "cacheblend_retrieve_prefetch_initial",
            category="blender",
            device="CPU->GPU",
            request_id=kwargs.get("req_id"),
            metadata={"tokens": len(tokens)},
        ):
            next(layerwise_retriever)
        yield

        for i in range(self.num_layers):
            with cf_span(
                "cacheblend_retrieve_layer_step",
                category="blender",
                device="CPU->GPU",
                request_id=kwargs.get("req_id"),
                layer_id=i,
                metadata={"tokens": len(tokens)},
            ):
                next(layerwise_retriever)
            with cf_span(
                "layerwise_model_recompute_and_repair_layer",
                category="blender",
                device="GPU",
                request_id=kwargs.get("req_id"),
                layer_id=i,
                metadata={"tokens": len(tokens)},
            ):
                next(layerwise_model_executor)
            yield

        with cf_span(
            "cacheblend_retrieve_finalize",
            category="blender",
            device="CPU->GPU",
            request_id=kwargs.get("req_id"),
            metadata={"tokens": len(tokens)},
        ):
            next(layerwise_retriever)

        self.metadata.clean()
        yield

    def blend(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Perform blending for the given tokens.
        """

        if isinstance(tokens, list):
            tokens = torch.tensor(tokens).to(torch_device_type)

        layerwise_blender = self.blend_layer(tokens, mask, **kwargs)

        for i in range(self.num_layers + 2):
            next(layerwise_blender)
