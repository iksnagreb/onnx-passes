from onnx_passes.passes._base import Transformation, Sequential

from onnx_passes.passes._unbroadcast.elementwise import \
    UnbroadcastElementwise_v1
from onnx_passes.passes._eliminate.identity import EliminateIdentityExpand_v1


class UnbroadcastLoop_v1(Sequential, Transformation):
    """Exhaustively apply unbroadcasting transformations."""

    passes = [
        UnbroadcastElementwise_v1,
        EliminateIdentityExpand_v1
    ]

    exhaustive = True
