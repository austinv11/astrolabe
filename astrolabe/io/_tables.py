"""Safe, reversible result-table encoding for HDF5 (no pickle)."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as ipc


def _arrow(frame):
    sink = pa.BufferOutputStream()
    table = pa.Table.from_pandas(frame, preserve_index=False)
    with ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return np.frombuffer(sink.getvalue().to_pybytes(), dtype=np.uint8).copy()


def _read(data):
    return ipc.open_stream(pa.py_buffer(np.asarray(data, dtype=np.uint8))).read_all().to_pandas()


def _axis(index):
    levels = list(index.levels) if isinstance(index, pd.MultiIndex) else [index]
    return {
        "multi": isinstance(index, pd.MultiIndex),
        "names": json.dumps(list(index.names)),
        "levels": {str(i): _arrow(pd.DataFrame({"value": level})) for i, level in enumerate(levels)},
        "codes": {str(i): np.asarray(code) for i, code in enumerate(index.codes)}
        if isinstance(index, pd.MultiIndex) else {},
    }


def _restore_axis(value):
    names = json.loads(value["names"])
    levels = [pd.Index(_read(value["levels"][str(i)])["value"]) for i in range(len(value["levels"]))]
    if value["multi"]:
        return pd.MultiIndex(levels=levels, codes=[value["codes"][str(i)] for i in range(len(levels))], names=names)
    return levels[0].rename(names[0])


def encode_tables(obj):
    # GeoDataFrame has its own geometry-aware IO registration.
    if isinstance(obj, pd.DataFrame) and not hasattr(obj, "geometry"):
        neutral = obj.copy(deep=False)
        neutral.index = pd.RangeIndex(len(obj))
        neutral.columns = [f"c{i}" for i in range(obj.shape[1])]
        if not obj.shape[1]:
            neutral = pd.DataFrame({"empty_rows": np.zeros(len(obj), dtype=np.uint8)})
        return {"__astrolabe_type": "ResultTable", "data": _arrow(neutral),
                "index": _axis(obj.index), "columns": _axis(obj.columns),
                "attrs": json.dumps(obj.attrs), "empty_columns": not obj.shape[1]}
    if isinstance(obj, dict):
        return {key: encode_tables(value) for key, value in obj.items()}
    return obj


def decode_tables(obj):
    if isinstance(obj, dict):
        if obj.get("__astrolabe_type") == "ResultTable":
            frame = _read(obj["data"])
            if obj.get("empty_columns", False):
                frame = frame.iloc[:, :0]
            frame.index, frame.columns = _restore_axis(obj["index"]), _restore_axis(obj["columns"])
            frame.attrs.update(json.loads(obj["attrs"]))
            return frame
        return {key: decode_tables(value) for key, value in obj.items()}
    return obj
