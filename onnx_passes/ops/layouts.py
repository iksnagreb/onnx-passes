# Custom operator function registry
from onnx_passes.ops import register, op

# ONNX list attributes must be annotated as Sequence in ONNX Script
from typing import Sequence


# Converts between data layouts: Syntactically this acts just as a transpose but
# allows to attach some semantics via the assumes attribute (which otherwise is
# ignored by the operator). The LayoutConverter itself does not interact with
# the usual streamlining flow and can thus be used to demarcate sections of the
# graph operating on different data layouts. Intended usage is for switching
# between channels-first and channels-last layout of image-like data, or as a
# marker for custom checks or graph-surgery transformations.
@register
def LayoutConverter(x, assumes: Sequence[str], perm: Sequence[int]):
    return op.Transpose(x, perm=perm)
