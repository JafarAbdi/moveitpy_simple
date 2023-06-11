"""Substitution that can load xacro file with mappings involving any subsititutable."""

from collections.abc import Iterable
from pathlib import Path

from launch.frontend import expose_substitution
from launch.launch_context import LaunchContext
from launch.some_substitutions_type import SomeSubstitutionsType
from launch.substitution import Substitution
from launch.utilities import normalize_to_list_of_substitutions
from launch_param_builder import load_xacro


@expose_substitution("xacro")
class Xacro(Substitution):
    """Substitution that can access load xacro file with mappings involving any subsititutable."""

    def __init__(
        self,
        file_path: SomeSubstitutionsType,
        *,
        mappings: dict[SomeSubstitutionsType, SomeSubstitutionsType] | None = None,
    ) -> None:
        """Create a Xacro substitution."""
        super().__init__()

        self.__file_path = normalize_to_list_of_substitutions(file_path)
        self.__mappings = {} if mappings is None else mappings

    @classmethod
    def parse(cls, data: Iterable[SomeSubstitutionsType]):  # noqa: ANN206, ANN102
        """Parse `XacroSubstitution` substitution."""
        if len(data) != 1:
            msg = "xacro substitution expects only support one argument use 'command' subsititutoin for parsing args"
            raise TypeError(
                msg,
            )
        kwargs = {"file_path": data[0]}
        return cls, kwargs

    @property
    def file_path(self) -> list[Substitution]:
        """Getter for file_path."""
        return self.__file_path

    @property
    def mappings(self) -> dict[SomeSubstitutionsType, SomeSubstitutionsType]:
        """Getter for mappings."""
        return self.__mappings

    def describe(self) -> str:
        """Return a description of this substitution as a string."""
        mappings_formatted = ", ".join(
            [f"{k.describe()}:={v.describe()}" for k, v in self.mappings.items()],
        )
        return f"Xacro(file_path = {self.file_path}, mappings = {mappings_formatted})"

    def perform(self, context: LaunchContext) -> str:
        """Perform the substitution by retrieving the mappings and context."""
        from launch.utilities import perform_substitutions

        expanded_file_path = perform_substitutions(context, self.__file_path)
        expanded_mappings = {}
        for key, value in self.__mappings.items():
            normalized_key = normalize_to_list_of_substitutions(key)
            normalized_value = normalize_to_list_of_substitutions(value)
            expanded_mappings[
                perform_substitutions(context, normalized_key)
            ] = perform_substitutions(context, normalized_value)

        return load_xacro(Path(expanded_file_path), mappings=expanded_mappings)
