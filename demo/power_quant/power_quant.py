# ir.Value, ir.conve
import onnx_ir as ir

# All algebraic passes are transformations derived from pattern-based rewrite
# rules
from onnx_passes.passes.base import Transformation, RewriteRulePass

# Need to import the passes module to set up the registry and make the
# @passes.register decorator work
import onnx_passes.passes as passes

# PowerQuantMatMul is defined in the custom domain and needs to be made
# available as an ONNX Script function once used
from onnx_passes.ops import DOMAIN as CUSTOM_DOMAIN

# NumPy used during match condition checks to operate on shapes and tensors
import numpy as np


@passes.verify.tolerance
@passes.register()
class FusePowerQuant(Transformation, RewriteRulePass):
    """Extracts the fused PowerQuantMatMul from the PowerQuant pattern."""

    @property
    def commute(self) -> bool:
        return True

    def pattern(self, op, x, w, b, alpha, alpha_inverse, sx, sw, min_w, max_w):
        """Match MatMul with PowerQuant quantizers at the inputs."""
        return op.MatMul(
            # Input dequantization: Scale, power and bias (instead of Abs-Sign
            # to have positive inputs to the power, we bias the input to be
            # positive and bias back after dequantization).
            op.Add(
                op.Pow(
                    op.Mul(
                        x,
                        sx
                    ),
                    alpha_inverse
                ),
                b,
            ),
            # Weight quantization and dequantization: Matches the entire pattern
            # to extract and reorder the dequantization part and reinsert the
            # quantization part.
            op.Mul(  # noqa: Duplicate, see reinserted pattern below...
                op.Sign(w),
                op.Pow(
                    op.Mul(
                        sw,
                        op.Abs(
                            op.Clip(
                                op.Round(
                                    op.Div(
                                        op.Mul(
                                            op.Sign(w),
                                            op.Pow(
                                                op.Abs(w),
                                                alpha
                                            )
                                        ),
                                        sw
                                    )
                                ),
                                min_w,
                                max_w,
                            )
                        )
                    ),
                    alpha_inverse
                ),
            )
        )

    def check(self, op, x, w, b, alpha, alpha_inverse, sx, sw, min_w, max_w):
        """Check whether this is a valid PowerQuant for transformation."""

        # The inverse power as part of the fused dequantizer must be a scalar
        # constant. We do not force alpha and alpha_inverse to be related.
        if (a := ir.convenience.get_const_tensor(alpha_inverse)) is not None:
            return np.prod(a.shape) == 1
        return False

    def rewrite(self, op, x, w, b, alpha, alpha_inverse, sx, sw, min_w, max_w):
        """Rewrite the fused and reordered PowerQuantMatMul pattern."""
        return op.Add(
            op.Mul(
                # Reorder to have the dequantization scales follow the matrix
                # multiplication.
                op.Pow(
                    op.Mul(
                        sx,
                        sw
                    ),
                    alpha_inverse
                ),
                # Extracted PowerQuant matrix multiplication: Dequantization is
                # absorbed into the custom operator, weight quantization is
                # reinserted into the graph
                op.Mul(
                    op.PowerQuantMatMul(
                        x,
                        op.Clip(
                            op.Round(
                                op.Div(
                                    op.Mul(
                                        op.Sign(w),
                                        op.Pow(
                                            op.Abs(w),
                                            alpha
                                        )
                                    ),
                                    sw
                                )
                            ),
                            min_w,
                            max_w
                        ),
                        op.Reciprocal(alpha_inverse),
                        _domain=CUSTOM_DOMAIN
                    ),
                    op.Constant(value_float=2 ** -23)
                )
            ),
            # Reorder the input dequantization bias to follow the matrix
            # multiplication: (x + b) @ w = (x @ w) + (b @ w)
            op.MatMul(
                # Expand the bias shape to match the fully-broadcast input shape
                # expected by the matrix multiplication.
                op.Expand(
                    b,
                    op.Shape(
                        op.Add(
                            op.Pow(
                                op.Mul(
                                    x,
                                    sx
                                ),
                                alpha_inverse
                            ),
                            b
                        )
                    ),
                ),
                # The bias is transformed by a regular matrix multiplication
                # with the quantized-dequantized weights - reinsert the full
                # weight pattern without modification.
                op.Mul(  # noqa: Duplicate, consider matching the output to use
                    # a s ashortcut for reinserting the pattern
                    op.Sign(w),
                    op.Pow(
                        op.Mul(
                            sw,
                            op.Abs(
                                op.Clip(
                                    op.Round(
                                        op.Div(
                                            op.Mul(
                                                op.Sign(w),
                                                op.Pow(
                                                    op.Abs(w),
                                                    alpha
                                                )
                                            ),
                                            sw
                                        )
                                    ),
                                    min_w,
                                    max_w
                                )
                            )
                        ),
                        alpha_inverse
                    )
                )
            )
        )
