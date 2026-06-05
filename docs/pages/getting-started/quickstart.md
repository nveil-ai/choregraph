# Quick Start

This guide walks through creating a simple data pipeline with Choregraph.

## 1. Create a Choregraph instance

```python
from choregraph import Choregraph

cg = Choregraph(workspace_path="./my_project")
```

The `workspace_path` is where Kedro project files will be generated.

## 2. Add an input data source

```python
cg.add_input(
    id="population",
    location="data/population.csv",
    format="CSV",
    label="World Population",
    visibility=True,
)
```

## 3. Add transform nodes

```python
from choregraph.parser import InputPortSpec

# Filter to rows where population > 1,000,000
cg.add_node(
    id="large_countries",
    type="filter_greater_than",
    input_ports=[
        InputPortSpec(name="df", source_ref="population"),
        InputPortSpec(name="column", value="population"),
        InputPortSpec(name="value", value="1000000"),
    ],
    label="Large Countries",
)

# Sort by population descending
cg.add_node(
    id="sorted",
    type="sort_values",
    input_ports=[
        InputPortSpec(name="df", source_ref="large_countries"),
        InputPortSpec(name="columns", value="population"),
        InputPortSpec(name="ascending", value="false"),
    ],
    label="Sorted by Population",
)
```

## 4. Run the pipeline

```python
cg.run()
```

## 5. Access results

```python
df = cg.get_dataset("sorted_result")
print(df.head(10))
```

## 6. Export to XML

Save the pipeline specification for later reuse:

```python
cg.export_to_xml("pipeline.xml")
```

## Loading from XML

Existing pipelines can be loaded directly:

```python
cg = Choregraph(xml_spec="pipeline.xml")
cg.run()
```

## Using as a context manager

```python
with Choregraph(xml_spec="pipeline.xml") as cg:
    cg.run()
    df = cg.get_dataset("sorted_result")
```

## Next steps

- Browse the [Transform Library](../reference/library/index.md) for all available operations
- Read the [Architecture Overview](../architecture/index.md) to understand the pipeline flow
- See the [Developer Guide](../developer/index.md) to add custom transforms
