# SPDX-FileCopyrightText: 2026 NVEIL SAS
# SPDX-FileContributor: Clément Baraille
# SPDX-FileContributor: Pierre Jacquet
# SPDX-License-Identifier: AGPL-3.0-or-later

import unittest
from lxml import etree
import pandas as pd
from kedro.io import DataCatalog, MemoryDataset
from choregraph.connectors import DiveConnector

class MockInput:
    def __init__(self, id, location):
        self.id = id
        self.location = location

class MockSpec:
    def __init__(self, inputs, nodes):
        self.inputs = inputs
        self.nodes = nodes
    def get_name(self, id):
        return "mydata"

class MockCG:
    def __init__(self, spec):
        self.spec = spec
        self.workspace_path = None

class TestLocationRepro(unittest.TestCase):
    def test_location_with_spec_no_nodes(self):
        # Setup
        df = pd.DataFrame({"a": [1]})
        ds = MemoryDataset(df)
        cat = DataCatalog({"mydata": ds})
        
        # Mocking cg.spec on catalog
        input_file_path = "/abs/path/to/data.csv"
        spec = MockSpec(
            inputs=[MockInput("1", input_file_path)],
            nodes=[] # No nodes/transformations
        )
        cat.cg = MockCG(spec)
        
        # Execute
        conn = DiveConnector(cat)
        xml = conn.generate_visuspec_xml()
        
        # Assert
        root = etree.fromstring(xml.encode("utf-8"))
        raw = root.find("rawData")
        file_elem = raw.find("file")
        location = file_elem.get("location")
        
        print(f"Location found: {location}")
        
        # DESIRED BEHAVIOR: matches "choregraph.xml" (default_location)
        self.assertEqual(location, "choregraph.xml")

if __name__ == '__main__':
    unittest.main()
