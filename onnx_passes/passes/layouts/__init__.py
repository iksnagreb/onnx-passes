from onnx_passes.ops import DOMAIN as CUSTOM_DOMAIN


class LayoutError(Exception):
    ...


def layout_converter_pattern(op):
    """Pattern matching layout converter."""
    return op.LayoutConverter(_allow_other_inputs=True, _domain=CUSTOM_DOMAIN)
