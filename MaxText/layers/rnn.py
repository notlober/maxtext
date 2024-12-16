"""Recurrent Layer."""

import functools
from typing import Any, Optional

from flax import linen as nn
import jax
from jax.ad_checkpoint import checkpoint_name
import jax.numpy as jnp

import common_types
from layers import linears
from layers import quantizations
from layers import initializers

Array = common_types.Array
Config = common_types.Config
DType = common_types.DType
Mesh = common_types.Mesh

DenseGeneral = linears.DenseGeneral
Quant = quantizations.AqtQuantization
nd_dense_init = initializers.nd_dense_init

class RNN(nn.Module):
    """recurrent layer implementation.
    
    Attributes:
        config: Model configuration
        mesh: Device mesh for sharding
        dtype: Computation dtype
        weight_dtype: Weight storage dtype
        quant: Optional quantization parameters
    """
    
    config: Config
    mesh: Mesh
    dtype: DType = jnp.float32
    weight_dtype: DType = jnp.float32
    quant: Optional[Quant] = None

    @nn.compact
    def __call__(
        self,
        inputs_q: Array,
        inputs_kv: Array,
        inputs_positions: Array,
        decoder_segment_ids: Array | None = None,
        *,
        model_mode: str = common_types.MODEL_MODE_TRAIN,
        deterministic: bool = False,
    ):
        """Applies recurrent processing on the input data."""
        
        hidden_dim = inputs_q.shape[-1] * 2
        
        # Linear1: project to larger dimension using proper initialization
        linear1 = DenseGeneral(
            features=hidden_dim,
            axis=-1,
            kernel_init=nd_dense_init(1.0, "fan_in", "normal"),
            kernel_axes=("embed", "hidden"),
            dtype=self.dtype,
            weight_dtype=self.weight_dtype,
            name="linear1",
            quant=self.quant,
            matmul_precision=self.config.matmul_precision,
        )
        
        Z = linear1(inputs_q)
        Z = checkpoint_name(Z, "linear1_output")

        # Cumulative sum along sequence dimension
        H_before_norm = jax.lax.cumsum(Z, axis=1)
        H_before_norm = checkpoint_name(H_before_norm, "cumsum_output")
        
        # Normalize - compute norm and divide
        norms = jnp.linalg.norm(H_before_norm, axis=-1, keepdims=True)
        H_normalized = H_before_norm / (norms + 1e-5)  # Add epsilon for stability
        H_normalized = checkpoint_name(H_normalized, "normalized_output")

        # Linear2: project back to original dimension using proper initialization
        linear2 = DenseGeneral(
            features=inputs_q.shape[-1],
            axis=-1, 
            kernel_init=nd_dense_init(1.0, "fan_in", "normal"),
            kernel_axes=("hidden", "embed"),
            dtype=self.dtype,
            weight_dtype=self.weight_dtype,
            name="linear2",
            quant=self.quant,
            matmul_precision=self.config.matmul_precision,
        )

        output = linear2(H_normalized)
        output = checkpoint_name(output, "linear2_output")

        # Optional dropout
        if not deterministic and self.config.dropout_rate > 0:
            output = nn.Dropout(
                rate=self.config.dropout_rate,
                broadcast_dims=(1,)  # Apply along sequence dimension
            )(output, deterministic=deterministic)
            
        return output
