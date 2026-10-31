from onnx_passes.passes.base import Sequential

from onnx_passes.passes.unbroadcast.elementwise import UnbroadcastElementwise_v1
from onnx_passes.passes.eliminate.identity import EliminateIdentityExpand_v1


class UnbroadcastLoop_v1(Sequential):
    """Exhaustively apply unbroadcasting transformations."""

    passes = [
        UnbroadcastElementwise_v1,
        EliminateIdentityExpand_v1
    ]

    exhaustive = True
