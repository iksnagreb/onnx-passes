from onnx_passes.passes._base import RewriteRule
from onnx_passes.passes._verify import Verify

from onnx_passes.traits.elementwise import produced_by_elementwise

import onnx_ir as ir


def produced_by_where(_, value: ir.Value) -> bool:
    """Check whether value is produced by a Where operation."""
    return (node := value.producer()) is not None and node.op_type == "Where"


class MoveWherePastElementwise_v1(RewriteRule, Verify):
    """Reorder Where distributing following elementwise into both branches.

    Rewrites n-ary elementwise f(x1, x2, ..., condition ? lhs : rhs, ..., xn)
    as condition ? f(x1, x2, ..., lhs, ..., xn) : f(x1, x2, ..., rhs, ..., xn).

    Does not apply to all boolean inputs as these should be normalized to pure
    boolean expressions via RewriteBooleanWhereAsXor_v1
    """

    @staticmethod
    def pattern(op):
        return op.submodule("")(_allow_other_inputs=True, _outputs=["out"])

    @staticmethod
    def check(context, out):
        if not produced_by_elementwise(context, out):
            return False

        if produced_by_where(context, out):
            return False

        if out.shape is None or out.shape.is_dynamic():
            return False

        # As soon as any input is produced by a not constant Where (handled by
        # constant folding) with non-boolean inputs (handled as a normalization
        # to Xor), accept this for reordering.
        for x in out.producer().inputs:
            if x.dtype != ir.DataType.BOOL:
                if produced_by_where(context, x):
                    if any(ir.convenience.get_const_tensor(v) is None
                           for v in x.producer().inputs):
                        return True

        return False

    @staticmethod
    def rewrite(op, out):
        # Find the elementwise operator which produces the matched value (the
        # value level check guarantees this exists and is indeed the node we are
        # interested in).
        elementwise = out.producer()

        # Collect the list of inputs to the elementwise operation with inputs
        # separated into a lhs and rhs branch, which differ exactly in the place
        # the Where operator is extracted: Rewire the lhs and rhs of the Where
        # into each input list and extract the condition input.
        condition, lhs, rhs = None, [], []

        for inp in elementwise.inputs:
            # Extract only the first instance of Where among the elementwise
            # inputs, reinsert each following as is.
            if condition is None and produced_by_where(None, inp):
                if all(ir.convenience.get_const_tensor(v) is not None
                       for v in inp.producer().inputs):
                    lhs.append(inp)
                    rhs.append(inp)

                    continue

                condition, x, y = inp.producer().inputs

                lhs.append(x)
                rhs.append(y)
            else:
                lhs.append(inp)
                rhs.append(inp)

        # Insert the replacement pattern with attributes transplanted from the
        # elementwise operator into each branch.
        return op.Where(
            condition,
            op.op(
                elementwise.op_type, *lhs, **elementwise.attributes
            ),
            op.op(
                elementwise.op_type, *rhs, **elementwise.attributes
            )
        )
