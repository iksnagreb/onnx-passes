from onnx_passes.passes._config import Config
from onnx_passes.passes._base import Sequential, VerificationError

from pathlib import Path
from typing import Optional

import inspect
import pydantic_settings

import onnx_ir as ir

# Setup pretty printing of messages and tracebacks (registered to automatically
# format exceptions)
import rich.traceback
import rich.console

console = rich.console.Console()
rich.traceback.install(console=console)


class OnnxPasses(Config, use_attribute_docstrings=True):
    """ONNX Passes: Model graph transformation and analysis."""

    model: pydantic_settings.CliPositionalArg[Path]
    """Path to ONNX model file"""

    passes: pydantic_settings.CliPositionalArg[list[str]] = []
    """List of passes to apply: resolvable name or module"""

    browse: pydantic_settings.CliToggleFlag[bool] = False
    """Browse the output model (or checkpoint after fail) in netron"""

    output: Optional[Path] = None
    """Path to transformed ONNX model output file"""

    exhaustive: pydantic_settings.CliToggleFlag[bool] = False
    """Keep applying passes exhaustively until the model stops changing"""

    unknown_args: pydantic_settings.CliUnknownArgs

    model_config = pydantic_settings.SettingsConfigDict(
        cli_shortcuts={
            "output": "o",
            "logging.verbose": ["verbose", "v"],
            "logging.checkpoint": "checkpoint",
            "logging.keep-intermediates": "keep-intermediates"
        },
        cli_kebab_case=True,
        cli_avoid_json=True,
        cli_ignore_unknown_args=True,
        cli_hide_none_type=True,
        nested_model_default_partial_update=True,
        # Load settings from YAML file if present. Can be overridden by CLI.
        yaml_file="passes.yaml",
        # Allow extra settings which could be picked up and managed by
        # individual passes
        extra="allow"
    )

    @classmethod
    def settings_customise_sources(
            cls,
            settings_cls: type[pydantic_settings.BaseSettings],
            init_settings: pydantic_settings.PydanticBaseSettingsSource,
            env_settings: pydantic_settings.PydanticBaseSettingsSource,
            dotenv_settings: pydantic_settings.PydanticBaseSettingsSource,
            file_secret_settings: pydantic_settings.PydanticBaseSettingsSource,
    ) -> tuple[pydantic_settings.PydanticBaseSettingsSource, ...]:
        return (
            pydantic_settings.YamlConfigSettingsSource(
                settings_cls, deep_merge=False
            ),
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings
        )

    def cli_cmd(self) -> None:
        # Set up a local pass sequence resolving passes from the list and
        # directly apply the sequence to the model loaded as ONNX IR
        class Passes_v0(Sequential):
            passes = self.passes
            exhaustive = self.exhaustive

        try:
            with console.status(
                    f"Applying passes to {self.model}...", spinner="line"
            ):
                result = Passes_v0(self).call(ir.load(self.model))
        except Exception as error:
            while error.__cause__ is not None:
                error = error.__cause__

            # Try to extract the model from the failed context to save model and
            # state for debugging and drop into netron if browsing output
            try:
                model = inspect.trace()[-1][0].f_locals["model"]

                import pickle
                import lzma

                ir.save(model, filename := "debug-checkpoint.onnx")

                if state := model.meta["passes"]:
                    with lzma.open(f"{filename}.passes.pkl.xz", "wb") as file:
                        pickle.dump(state, file)  # noqa: lzma file
            except:  # noqa: broad on purpose
                filename = None

            if isinstance(error, VerificationError):
                console.print(f"{error.__class__.__name__}: {error}")
            else:
                console.print_exception()

            if self.browse and filename:
                try:
                    import netron
                except ModuleNotFoundError:
                    raise ModuleNotFoundError(
                        "Install netron to browse the debug checkpoint"
                    ) from None

                netron.serve(filename, None, browse=True)  # noqa: data=None
                netron.wait()

            return

        # If configured save the result model and serialize the state metadata
        # as a compressed pickle file and dump the settings as applied
        if self.output:
            import pickle
            import lzma
            import yaml

            ir.save(result.model, self.output)

            if state := result.model.meta["passes"]:
                with lzma.open(f"{self.output}.passes.pkl.xz", "wb") as file:
                    pickle.dump(state, file)  # noqa: 'SupportsWrite[bytes]'?

                # Update settings with resolved passes which actually modified
                # the model to save these to the settings dump
                self.passes = [
                    f"{p.identifier}" for p in state.modified_by
                ]

            with open(f"{self.output}.passes.yaml", "w") as file:
                yaml.safe_dump(
                    self.model_dump(mode="json", exclude_defaults=True), file
                )

        if state := result.model.meta["passes"]:
            # Collect and print basic pass application statistics from the state
            # dictionary tracked via model metadata
            total, modified_by, verified_by = (
                state.counter, len(state.modified_by), len(state.verified_by)
            )

            console.print(
                f"Applied {total} passes (model modified by {modified_by},"
                f" verified {verified_by})"
            )

        # If configured browse the output model in netron, either serving the
        # output file is saved or from serialized proto
        if self.browse:
            try:
                import netron
            except ModuleNotFoundError:
                raise ModuleNotFoundError(
                    "Install netron to browse the result"
                ) from None

            if filename := self.output:
                data = None
                filename = str(filename)
            else:
                data = ir.to_proto(result.model).SerializeToString()
                filename = "result.model.onnx"

            netron.serve(filename, data, browse=True)  # noqa: data=None
            netron.wait()


def main():
    pydantic_settings.CliApp.run(OnnxPasses)


if __name__ == "__main__":
    main()
