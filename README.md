<p align="center">
  <img src="https://raw.githubusercontent.com/nveil-ai/nveil-toolkit/main/assets/logo.png" alt="NVEIL" width="140">
</p>

<h1 align="center">Choregraph</h1>

<p align="center">
  <strong>The pure-Python data-processing engine behind NVEIL — usable on its own.</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/choregraph/"><img src="https://img.shields.io/pypi/v/choregraph?color=orange&label=PyPI" alt="PyPI"></a>
  <a href="https://pypi.org/project/choregraph/"><img src="https://img.shields.io/pypi/pyversions/choregraph?color=blue" alt="Python"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-AGPL--3.0--or--later-blue" alt="License"></a>
  <a href="https://docs.nveil.com"><img src="https://img.shields.io/badge/docs-docs.nveil.com-blue" alt="Docs"></a>
</p>

---

Choregraph turns declarative pipeline specifications into executable [Kedro](https://kedro.org/) data pipelines. It is the data-processing engine inside the [NVEIL platform](https://github.com/nveil-ai/nveil) — and a **standalone library you can use on its own**: wire inputs and transforms in Python (or portable XML), run them locally, and get DataFrames back. No NVEIL server required.

## Install

```bash
pip install choregraph
```

Optional extras:

```bash
pip install "choregraph[extra]"    # geo (geopandas), NLP, scikit-learn, Pillow, …
pip install "choregraph[dicom]"    # DICOM medical imaging
```

## Quick start

```python
from choregraph import Choregraph
from choregraph.parser import InputPortSpec

cg = Choregraph(workspace_path="./my_project")
cg.add_input(id="population", location="data/population.csv", format="CSV")

cg.add_node(
    id="large",
    type="filter_greater_than",
    input_ports=[
        InputPortSpec(name="df", source_ref="population"),
        InputPortSpec(name="column", value="population"),
        InputPortSpec(name="value", value="1000000"),
    ],
)

cg.run()
df = cg.get_dataset("large_result")
```

Pipelines are portable — save and reload them as XML:

```python
cg.export_to_xml("pipeline.xml")

with Choregraph(xml_spec="pipeline.xml") as cg:
    cg.run()
    df = cg.get_dataset("large_result")
```

Full walkthrough → **[docs.nveil.com](https://docs.nveil.com)**.

## What's inside

- **Declarative pipelines** — define inputs, transforms, and outputs in Python or portable XML, executed as Kedro pipelines.
- **50+ transforms** — filtering, aggregation, joins, pivots, normalization, discretization, row/column operations, …
- **Geo** — geocode location names to coordinates; join country-boundary polygons for map visualizations.
- **NLP** — multi-label binarization with automatic language detection, lemmatization, and fuzzy matching.
- **Excel intelligence** — LLM-assisted detection and tidying of messy multi-table spreadsheets.
- **Biosignals** — EDF/EDF+ ingestion (EEG, ECG, polysomnography, …).

## Part of NVEIL

Choregraph is one of the open-source engines of the **[NVEIL platform](https://github.com/nveil-ai/nveil)**. Most people reach it through the [NVEIL Toolkit](https://github.com/nveil-ai/nveil-toolkit) (`pip install nveil`) or the platform itself — but Choregraph stands on its own for anyone who wants a programmable, XML-portable data-processing engine. Its results export straight to [DIVE](https://github.com/nveil-ai/dive) for visualization.

## Contributing

Contributions are welcome under the project's **[Contributor License Agreement](CLA.md)** — signed once, on your first pull request (a license grant, not an assignment; you keep your rights). Bug reports and ideas are welcome via [GitHub Issues](https://github.com/nveil-ai/choregraph/issues).

## License

Dual-licensed: **AGPL-3.0-or-later** (see [`LICENSE`](LICENSE)) and a **commercial** license for closed-source / proprietary use (see [`COMMERCIAL-LICENSE.md`](COMMERCIAL-LICENSE.md)) — contact `pierre.jacquet@nveil.com`.
