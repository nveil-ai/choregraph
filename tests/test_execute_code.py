# SPDX-FileCopyrightText: 2026 NVEIL SAS
# SPDX-FileContributor: Pierre Jacquet
# SPDX-FileContributor: Guillaume Franque
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Tests for the execute_code transform and _validate_code_safety."""
from __future__ import annotations

import pytest
import pandas as pd
import numpy as np

from choregraph.library import execute_code, _validate_code_safety


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def simple_df() -> pd.DataFrame:
    return pd.DataFrame({"x": [1, 5, 10, 15], "y": [10, 20, 30, 40]})


# ---------------------------------------------------------------------------
# Valid code execution
# ---------------------------------------------------------------------------

class TestExecuteCode:
    def test_basic_filter(self, simple_df):
        result = execute_code(code="result = df[df['x'] > 5]", df=simple_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 2
        assert list(result["x"]) == [10, 15]

    def test_multiline_code(self, simple_df):
        code = (
            "filtered = df[df['x'] > 1]\n"
            "result = filtered.assign(z=filtered['x'] * 2)"
        )
        result = execute_code(code=code, df=simple_df)
        assert "z" in result.columns
        assert list(result["z"]) == [10, 20, 30]

    def test_numpy_available(self, simple_df):
        result = execute_code(code="result = df.assign(log_x=np.log(df['x']))", df=simple_df)
        assert "log_x" in result.columns

    def test_allowed_import(self, simple_df):
        """scipy should be importable (not blocked)."""
        _validate_code_safety("from scipy.stats import zscore")  # Should not raise

    def test_series_auto_converted(self, simple_df):
        result = execute_code(code="result = df['x']", df=simple_df)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 4

    def test_multi_input(self):
        sales = pd.DataFrame({"region_id": [1, 2], "revenue": [100, 200]})
        regions = pd.DataFrame({"region_id": [1, 2], "name": ["North", "South"]})
        result = execute_code(
            code="result = pd.merge(sales, regions, on='region_id')",
            sales=sales,
            regions=regions,
        )
        assert "name" in result.columns
        assert len(result) == 2

    def test_input_not_mutated(self, simple_df):
        original = simple_df.copy()
        execute_code(code="df['x'] = 999; result = df", df=simple_df)
        pd.testing.assert_frame_equal(simple_df, original)

    def test_nested_function_can_access_pd_and_inputs(self):
        """Functions defined inside code must see pd, np, and input DataFrames."""
        df = pd.DataFrame({"a": [1, 2, 3]})
        code = (
            "def transform(data):\n"
            "    return pd.DataFrame({'b': data['a'] * np.int64(2)})\n"
            "result = transform(df)\n"
        )
        result = execute_code(code=code, df=df)
        assert list(result["b"]) == [2, 4, 6]


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

class TestExecuteCodeErrors:
    def test_missing_result_variable(self, simple_df):
        with pytest.raises(ValueError, match="must assign output to a variable named 'result'"):
            execute_code(code="x = df.head()", df=simple_df)

    def test_non_dataframe_result(self, simple_df):
        with pytest.raises(TypeError, match="must be a DataFrame"):
            execute_code(code="result = 42", df=simple_df)


# ---------------------------------------------------------------------------
# Safety validation
# ---------------------------------------------------------------------------

class TestValidateCodeSafety:
    def test_blocked_import_os(self):
        with pytest.raises(ValueError, match="Import of 'os' is not allowed"):
            _validate_code_safety("import os")

    def test_blocked_import_subprocess(self):
        with pytest.raises(ValueError, match="Import of 'subprocess' is not allowed"):
            _validate_code_safety("import subprocess")

    def test_blocked_import_from_socket(self):
        with pytest.raises(ValueError, match="Import from 'socket' is not allowed"):
            _validate_code_safety("from socket import create_connection")

    def test_blocked_import_nested(self):
        with pytest.raises(ValueError, match="Import of 'os.path' is not allowed"):
            _validate_code_safety("import os.path")

    def test_blocked_call_exec(self):
        with pytest.raises(ValueError, match="Call to 'exec' is not allowed"):
            _validate_code_safety("exec('print(1)')")

    def test_blocked_call_eval(self):
        with pytest.raises(ValueError, match="Call to 'eval' is not allowed"):
            _validate_code_safety("eval('1+1')")

    def test_blocked_call_open(self):
        with pytest.raises(ValueError, match="Call to 'open' is not allowed"):
            _validate_code_safety("f = open('/etc/passwd')")

    def test_blocked_attr_subclasses(self):
        with pytest.raises(ValueError, match="Access to '__subclasses__' is not allowed"):
            _validate_code_safety("x.__class__.__subclasses__()")

    def test_blocked_attr_class(self):
        with pytest.raises(ValueError, match="Access to '__class__' is not allowed"):
            _validate_code_safety("x.__class__")

    def test_blocked_attr_globals(self):
        with pytest.raises(ValueError, match="Access to '__globals__' is not allowed"):
            _validate_code_safety("func.__globals__")

    def test_blocked_attr_builtins(self):
        with pytest.raises(ValueError, match="Access to '__builtins__' is not allowed"):
            _validate_code_safety("x.__builtins__")

    def test_code_too_long(self):
        with pytest.raises(ValueError, match="Code exceeds maximum length"):
            _validate_code_safety("x = 1\n" * 2000)

    def test_syntax_error(self):
        with pytest.raises(ValueError, match="Invalid Python syntax"):
            _validate_code_safety("def (broken")

    def test_allowed_pandas_numpy(self):
        _validate_code_safety("import pandas; import numpy")

    def test_allowed_scipy(self):
        _validate_code_safety("from scipy.stats import zscore")

    def test_allowed_sklearn(self):
        _validate_code_safety("import sklearn.preprocessing")

# ---------------------------------------------------------------------------
# Registry integration
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_execute_code_in_registry(self):
        from choregraph.library import TRANSFORM_REGISTRY
        assert "execute_code" in TRANSFORM_REGISTRY
        assert TRANSFORM_REGISTRY["execute_code"]["func"] is execute_code
        assert TRANSFORM_REGISTRY["execute_code"]["output_type"] is pd.DataFrame
