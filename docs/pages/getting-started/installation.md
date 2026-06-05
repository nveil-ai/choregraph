# Installation

## Requirements

- Python **>= 3.10**
- pip

## Install from source

```bash
# Clone and install in editable mode
pip install -e ./choregraph

# With documentation tooling
pip install -e "./choregraph[docs]"
```

## Core Dependencies

Choregraph installs the following automatically:

| Package | Purpose |
|---------|---------|
| `kedro` | Pipeline orchestration framework |
| `kedro-datasets` | Data catalog and dataset types |
| `pandas` | DataFrame processing |
| `pyarrow` | Parquet serialization |
| `lxml` | XML parsing and generation |
| `geopandas` | Geospatial operations |
| `geonamescache` | City/country geocoding lookups |
| `langdetect`, `simplemma`, `unidecode` | NLP language detection and lemmatization |
| `rapidfuzz` | Fuzzy string matching for NLP hints |
