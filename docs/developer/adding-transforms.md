# Adding Transforms

This guide walks through adding a new transform function to Choregraph.

## 1. Write the function

Add your function to `choregraph/src/choregraph/library.py` (or a collection module for
domain-specific operations).

Follow the existing pattern — accept a DataFrame and parameters, return a DataFrame:

```python
def my_transform(df: pd.DataFrame, column: str, threshold: float = 0.5) -> pd.DataFrame:
    """Short description of what the transform does.

    Args:
        df: Input DataFrame.
        column: Name of the column to process.
        threshold: Cutoff value for the operation.

    Returns:
        DataFrame with the transformation applied.
    """
    result = df.copy()
    result[f"{column}_transformed"] = result[column].apply(lambda x: x > threshold)
    return result
```

### Conventions

- First parameter is typically `df: pd.DataFrame` (connected input)
- Static parameters (column names, thresholds, flags) come after
- Return a new DataFrame — avoid mutating the input
- Use Google-style docstrings with `Args:` and `Returns:` sections

## 2. Register in TRANSFORM_REGISTRY

At the bottom of `library.py`, add your function to the registry:

```python
TRANSFORM_REGISTRY = {
    # ... existing entries ...
    "my_transform": {"func": my_transform, "output_type": pd.DataFrame},
}
```

The key must match the `type` attribute used in XML node definitions.

## 3. Add XSD type definition

In `choregraph/src/choregraph/TransformGraph.xsd`, add a `complexType` defining your
function's ports:

```xml
<!-- My custom transform: applies threshold-based transformation -->
<xs:complexType name="my_transform">
    <xs:sequence>
        <xs:element name="inputPort" minOccurs="2" maxOccurs="3">
            <xs:complexType>
                <xs:attribute name="name" use="required">
                    <xs:simpleType>
                        <xs:restriction base="xs:string">
                            <xs:enumeration value="df"/>
                            <xs:enumeration value="column"/>
                            <xs:enumeration value="threshold"/>
                        </xs:restriction>
                    </xs:simpleType>
                </xs:attribute>
                <!-- Port type varies by name -->
            </xs:complexType>
        </xs:element>
        <xs:element name="outputPort" minOccurs="1" maxOccurs="1"/>
    </xs:sequence>
</xs:complexType>
```

The XSD definition enables:

- Parameter type conversion in the builder (string → float for `threshold`)
- Function catalogue generation via `xsd_catalogue_utils`
- Validation of XML specifications

## 4. Write tests

Add tests in `choregraph/tests/test_library.py`:

```python
def test_my_transform(sample_df):
    result = my_transform(sample_df, column="Score", threshold=80.0)
    assert "Score_transformed" in result.columns
    assert len(result) == len(sample_df)
```

## 5. Verify

```bash
# Run tests
pytest choregraph/tests/test_library.py -v

# Check that the function appears in the catalogue
python -c "from choregraph.library import TRANSFORM_REGISTRY; print('my_transform' in TRANSFORM_REGISTRY)"
```
