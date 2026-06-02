# API Reference

Complete reference for all Choregraph modules, classes, and functions.

## Core

The main classes that drive pipeline lifecycle:

- [Choregraph (Facade)](choregraph.md) — Primary entry point for pipeline management
- [Parser](parser.md) — XML specification parsing and dataclass model
- [Builder](builder.md) — Spec-to-Kedro pipeline construction
- [Wrapper](wrapper.md) — Kedro project file generation

## Transform Library

The extensible registry of data operations:

- [Core Transforms](library/core.md) — 50 filtering, aggregation, column/row, and advanced operations
- [Excel Transforms](library/excel.md) — LLM-assisted Excel tidying
- [Geo Collection](library/geo.md) — Geocoding and country boundary operations
- [NLP Collection](library/nlp.md) — Multi-label text binarization

## Infrastructure

Support modules for metadata, loading, execution, and visualization:

- [Metadata](metadata.md) — DataFrame field metadata extraction
- [Loaders](loaders.md) — CSV sniffing and load configuration
- [Hooks](hooks.md) — Kedro execution status tracking
- [Viz Server](viz.md) — Kedro Viz server management
- [XSD Catalogue](xsd-catalogue.md) — Function catalogue extraction from XSD

## Connectors

- [DIVE Connector](connectors/dive.md) — VisuSpec XML export for the DIVE kernel
