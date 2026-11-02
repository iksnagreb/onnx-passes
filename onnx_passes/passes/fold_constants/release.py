from onnx_passes.passes.base import Transformation, Sequential
from onnx_passes.passes.verify import Verify

import onnx_ir as ir


class ReleaseConstants_v1(Transformation, Verify):
    """Release constant tensors from constant stop metadata mark.

    This enables previously blocked constant operations to participate in
    constant folding again.
    """

    def call(self, model: ir.Model) -> ir.passes.PassResult:

        modified = False

        for _, x in ir.convenience.create_value_mapping(model.graph).items():
            if x.meta.get("constant_stop") is not None:
                x.meta.pop("constant_stop")
                modified = True

        return ir.passes.PassResult(model, modified)


from onnx_passes.passes import fold_constants


class FoldConstantsRelease_v1(Sequential):
    """Release constants and immediately apply constant folding."""

    passes = [
        ReleaseConstants_v1,
        fold_constants
    ]
