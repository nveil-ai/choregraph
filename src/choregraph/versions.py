# SPDX-FileCopyrightText: 2026 NVEIL SAS
# SPDX-FileContributor: Pierre Jacquet
# SPDX-License-Identifier: AGPL-3.0-or-later

"""TransformGraph schema version constants.

The schema version tracks the structure of choregraph XML. It is stamped
by the serializer — not carried through the AI processing pipeline.

See docs/versioning.md for the additive-only policy.
"""

TRANSFORMGRAPH_SCHEMA_VERSION = "1.0.0"
"""Current TransformGraph schema version, written by the serializer."""

TRANSFORMGRAPH_MIN_COMPATIBLE = "1.0.0"
"""Minimum schema version the parser can load without migration."""
