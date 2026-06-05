# Builder

Converts a `ChoregraphSpec` into a Kedro `Pipeline` object. Sits between the parser
(which produces the spec) and the Kedro runner. Resolves input/output port connections,
applies XSD-based type conversion for parameters, and generates Kedro node definitions.

::: choregraph.builder
    options:
      members_order: source
      show_source: true
