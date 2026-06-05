# Parser

Defines the data model for Choregraph pipeline specifications and provides XML parsing.

The dataclasses here (`ChoregraphSpec`, `InputSpec`, `NodeSpec`, `InputPortSpec`, `OutputPortSpec`)
form the in-memory representation of a pipeline graph. `ChoregraphSpecParser` reads XML into
these structures.

::: choregraph.parser
    options:
      members_order: source
      show_source: true
