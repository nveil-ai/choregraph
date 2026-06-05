# XML Specification

Choregraph pipelines are defined using an XML format validated against `TransformGraph.xsd`.

## Structure

A pipeline specification has three sections:

```xml
<choregraph>
    <inputs>
        <!-- Data source definitions -->
    </inputs>
    <pipeline>
        <!-- Transform node definitions -->
    </pipeline>
</choregraph>
```

## Inputs

Each `<input>` declares a data source:

```xml
<inputs>
    <input id="sales"
           label="Sales Data"
           location="data/sales.csv"
           format="CSV"
           visibility="true"
           fieldSeparator=","
           header="true" />

    <input id="regions"
           label="Region Lookup"
           location="data/regions.csv"
           format="CSV" />
</inputs>
```

| Attribute | Required | Description |
|-----------|----------|-------------|
| `id` | Yes | Unique identifier (referenced by nodes) |
| `label` | No | Human-readable name (auto-generated from ID if omitted) |
| `location` | Yes | File path or URL |
| `format` | Yes | Data format: `CSV`, `JSON`, `EXCEL` |
| `visibility` | No | Whether to expose in visualization (`true`/`false`) |
| `fieldSeparator` | No | CSV column delimiter (default: auto-detect) |
| `header` | No | Whether the CSV has a header row |
| `skipLines` | No | Number of lines to skip at the start |

## Nodes

Each `<node>` defines a transform operation:

```xml
<pipeline>
    <node id="top_sales"
          label="Top 10 Sales"
          type="get_top_n">
        <inputPort name="df" source_ref="sales" />
        <inputPort name="column" value="revenue" />
        <inputPort name="n" value="10" />
        <outputPort id="101"
                    name="result"
                    label="Top Sales"
                    visibility="true" />
    </node>

    <node id="summary"
          label="Revenue Summary"
          type="aggregate_sum">
        <inputPort name="df" source_ref="101" />
        <inputPort name="group_columns" value="region" />
        <outputPort id="102"
                    name="result"
                    label="Revenue by Region"
                    visibility="true" />
    </node>
</pipeline>
```

### Input Ports

Ports connect data or pass parameters to transform functions:

| Attribute | Description |
|-----------|-------------|
| `name` | Parameter name matching the Python function signature |
| `source_ref` | ID of the input or output port providing data (connected port) |
| `value` | Static parameter value as a string (converted to Python type by the builder) |

A port has either `source_ref` (connected) or `value` (static), not both.

### Output Ports

| Attribute | Description |
|-----------|-------------|
| `id` | Unique integer ID (used as `source_ref` by downstream nodes) |
| `name` | Port name (typically `result` or `mask`) |
| `label` | Human-readable label for visualization |
| `visibility` | Whether to expose this output in DIVE visualization |

## Type Conversion

The builder converts string `value` attributes to Python types using the XSD schema:

| XSD Port Type | Python Type | Example |
|---------------|-------------|---------|
| `StaticFloatPort` | `float` | `"3.14"` → `3.14` |
| `StaticIntegerPort` | `int` | `"10"` → `10` |
| `StaticBooleanPort` | `bool` | `"true"` → `True` |
| `StaticListPort` | `list` | `"a,b,c"` → `["a", "b", "c"]` |
| `StaticStringPort` | `str` | `"revenue"` → `"revenue"` |
| `ConnectedDataFramePort` | — | Resolved via `source_ref` |

## Programmatic Equivalent

The same pipeline can be built without XML:

```python
from choregraph import Choregraph
from choregraph.parser import InputPortSpec, OutputPortSpec

cg = Choregraph()
cg.add_input(id="sales", location="data/sales.csv", format="CSV")
cg.add_node(
    id="top_sales",
    type="get_top_n",
    input_ports=[
        InputPortSpec(name="df", source_ref="sales"),
        InputPortSpec(name="column", value="revenue"),
        InputPortSpec(name="n", value="10"),
    ],
    output_ports=[
        OutputPortSpec(id=101, name="result", label="Top Sales", visibility=True),
    ],
)
```
