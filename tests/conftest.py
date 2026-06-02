# SPDX-FileCopyrightText: 2025 NVEIL SAS
# SPDX-FileContributor: Pierre Jacquet
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest


# Ensure `src/` layout imports work when choregraph isn't installed editable.
_SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))


@pytest.fixture()
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Age": [10, 20, 30, 30],
            "Score": [1.5, 2.0, 3.25, 3.25],
            "Name": ["a", "b", "c", "c"],
        }
    )


@pytest.fixture()
def sample_xml_string(tmp_path: Path) -> str:
    # Minimal XML spec using the expected tags.
    csv_path = tmp_path / "people.csv"
    csv_path.write_text("Age,Score,Name\n10,1.5,a\n20,2.0,b\n", encoding="utf-8")

    return f"""<?xml version=\"1.0\" encoding=\"utf-8\"?>
<choregraph>
  <inputs>
    <input id=\"1\" location=\"{csv_path.as_posix()}\" format=\"CSV\" fieldSeparator=\",\" header=\"true\" />
  </inputs>
  <nodes>
    <node id=\"2\" type=\"select_columns\">
      <inputPorts>
        <port name=\"input\" sourceRef=\"1\" type=\"DATAFRAME\" />
        <port name=\"parameter\" value=\"Age\" type=\"PARAMETER\" />
      </inputPorts>
      <outputPorts>
        <port id=\"2_result\" name=\"result\" type=\"DATAFRAME\" visibility=\"true\" />
      </outputPorts>
    </node>
  </nodes>
</choregraph>
"""


@pytest.fixture()
def tmp_workspace(tmp_path: Path) -> Path:
    # Workspace layout used by Choregraph/ManagedProjectBuilder.
    # The library expects to create `pipeline` and `data/inputs` etc.
    ws = tmp_path / "workspace"
    ws.mkdir(parents=True, exist_ok=True)
    return ws
