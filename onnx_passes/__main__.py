# sys.exit for quitting upon error
import sys

# Dynamically import python modules at runtime used for dynamically registering
# passes according to configuration files
import importlib

# Click is used for building a command line interface from python functions as
# the entrypoint for the script
import click
# YAML is used for handling extra configuration options for the ONNX IR passes
import yaml
# Pickle is used for serializing the state dictionary tracked with the ONNX IR
# passes
import pickle

# Infrastructure for handling ONNX IR models and managing sequences of passes:
#   ir.load, ir.save, ir.to_proto, ir.PassManager
import onnx_ir as ir

# Collect custom ONNX IR passes from the library by name
from onnx_passes.passes import collect


# Main function called from the entrypoint below. Applies a sequence of ONNX IR
# passes to the model which is either saved to file or printed as raw proto
# representation to the standard output. Configuration options and state can be
# optionally loaded form files if specified.
#
# Wrapped as a command line interface configured by click: Function parameters
# are connected to command line options and arguments.
@click.command()
# Required positional arguments: A single input ONNX model must be specified
# followed by arbitrarily many ONNX IR passes referred to by (category) names
@click.argument("model", type=click.Path(exists=True))
@click.argument("passes", type=str, nargs=-1)
# Optional arguments specifying the output file and configuration and state
# dictionary to load before applying the ONNX IR passes
@click.option("-o", "output", type=click.Path(exists=False), default=None)
@click.option("-c", "config", type=click.Path(exists=True), default=None)
@click.option("-s", "state", type=click.Path(exists=True), default=None)
# TODO: Add some catch-all argument list collecting all unknown arguments to be
#  parsed and injected into the configuration dictionary...
def main(model: str, passes: list[str], output: str, config: str, state: str):
    # Initially assume empty configuration and state dictionaries shared by all
    # ONNX IR passes: these will be shared by reference!
    _config, _state = {}, {}

    # If a configuration file is specified, load the YAML into the configuration
    # dictionary
    if config is not None:
        with open(config, "r") as file:
            _config = yaml.safe_load(file)

    # If an initial state file is specified, load the pickle into the state
    # dictionary
    if state is not None:
        with open(state, "rb") as file:
            _state = pickle.load(file)

    # Inject dynamic module imports if the configuration specifies an imports
    # section, e.g., for dynamically registering passes
    if "imports" in _config:
        for name in _config["imports"]:
            importlib.__import__(name)

    # The configuration file can already specify an initial sequence of ONNX IR
    # passes which will be executed before any passes specified as arguments
    if "passes" in _config:
        passes = [*_config["passes"], *passes]

    # Collect and instantiate all ONNX IR passes from the sequence by name and
    # connect each pass to the shared configuration and state dictionary
    passes = [cls(_config, _state) for cls in collect(passes)]

    # Skip constructing and running the manager if there are no passes specified
    if passes:
        # Pass manager instance which repeatedly runs the sequence of passes on
        # the model and evaluates pre- and post-conditions of each pass, e.g.,
        # for automatic verification.
        passes = ir.passes.PassManager(passes=passes, steps=1)

        # Try to apply the sequence of manged passes until any exception, e.g.,
        # by verification occurs
        try:
            # Load ONNX IR to modify/analyze from the model file - format should
            # be inferred and apply the sequence of passes
            result = passes(ir.load(model))
            # If the composed pass is marked exhaustive, apply the sequence of
            # passes as long as there are changes to the model
            while _config.setdefault("exhaustive", False) and result.modified:
                result = passes(result.model)
        # Catch any exception and walk up the context chain to find the initial
        # causing exception
        except Exception as error:
            # Walk up until there is no further causing exception
            while error.__context__:
                # If we see some pre- or post-condition error along the way,
                # print this as well to see at which pass the error occurred
                if isinstance(error, ir.passes.PreconditionError):
                    print(f"{error.__class__.__name__}: {error}")
                if isinstance(error, ir.passes.PostconditionError):
                    print(f"{error.__class__.__name__}: {error}")
                # Continue walking up the chain of exceptions
                error = error.__context__
            # Make sure to always save the state dictionary if we have an output
            # file name - helps with debugging
            if output is not None:
                with open(f"{output}.pkl", "wb") as file:
                    pickle.dump(_state, file)  # noqa: 'SupportsWrite[bytes]'?
            # Exit with printing the exception name and message
            sys.exit(f"{error.__class__.__name__}: {error}")
    # Assume unmodified pass-through result in case there are no ONNX IR passes
    # specified
    else:
        result = ir.passes.PassResult(ir.load(model), False)

    # If an output file name is specified, serialize and save the output model
    # and the potentially modified state dictionary
    if output is not None:
        # Serialize the model back to the proto representations and save to the
        # specified file
        ir.save(result.model, output)
        # Derive the state dictionary name from the output model name by
        # appending the pickle suffix
        with open(f"{output}.pkl", "wb") as file:
            pickle.dump(_state, file)  # noqa: 'SupportsWrite[bytes]'?
    # If no output file name is specified, serialize the model back to the proto
    # representation but print to the standard output - state ist lost!
    else:
        print(ir.to_proto(result.model))


# Entry point when actually called as a main script and not just imported
if __name__ == "__main__":
    main()
