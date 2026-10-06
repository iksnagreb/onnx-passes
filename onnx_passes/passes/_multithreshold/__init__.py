from onnx_passes.passes._base import Sequential

from onnx_passes.passes import _reorder

from onnx_passes.passes._normalize import anf

from onnx_passes.passes._multithreshold import convert
from onnx_passes.passes._multithreshold import absorb
from onnx_passes.passes._multithreshold import normalize
from onnx_passes.passes._multithreshold import fuse


class ConvertToMultiThresholds_v1(Sequential):
    """Convert suitable subgraphs to MuliThreshold representation.

    The result of this pass is *not yet* a MultiThreshold custom operator, but
    rather the unnormalized/inlined representation based on ReduceSum.

    This representation is more suitable to be combined and looped with more
    generic optimizations, such as reordering and elimination passes, before
    normalizing the resulting subgraphs to MultiThreshold operators.
    """

    passes = [
        _reorder,
        convert,
        absorb
    ]

    exhaustive = True


class OptimizeMultiThresholds_v1(Sequential):
    """Optimize boolean subgraphs in algebraic normal form via reordering."""

    passes = [
        anf,
        _reorder,
        anf
    ]


class NormalizeMultiThresholds_v1(Sequential):
    """Normalize suitable subgraphs to optimized MultiThreshold operators.

    The resulting MultioThreshold operators are in their generalized form, i.e.,
    they can represent arbitrarily weighted/directed steps, as well as arbitrary
    parameter dimensions including multi-directional broadcasting.
    """

    passes = [
        normalize,
        fuse,
        _reorder
    ]

    exhaustive = True


class CompileToMultiThresholds_v1(Sequential):
    """Compile quantized activation paths to MultiThreshold functions.

    Composes a default pass sequence covering the most common compilation flow
    of converting, optimizing and normalizing MuliThreshold operators:

    1. Threshold conversion from Round and repeated, absorbing of functions into
       comparisons and reordering to enable more absorption passes.
    2. Threshold optimization in algebraic normal form (ANF) by repeated reorder
       passes and output normalization to ANF.
    3. Threshold normalization from ANF to fused MultiThreshold custom operator
       and specific optimizations: sorting, deduplication, elimination of dead
       thresholds and parameter tensor unbroadcasting.
    """

    passes = [
        ConvertToMultiThresholds_v1,
        OptimizeMultiThresholds_v1,
        NormalizeMultiThresholds_v1
    ]
