"""Window manifests and optional Unity Catalog Delta persistence."""

from pathlib import Path

from .audio import compute_quality_metrics, save_window

MANIFEST_FIELDS = (
    "session_id", "window_id", "audio_path", "start_offset_s", "end_offset_s",
    "duration_s", "sample_rate", "channels", "status", "peak_amplitude", "rms",
    "low_signal",
)


def build_window_manifest(session_id, windows, sample_rate, output_dir):
    """Write each waveform and return ordered metadata records for all windows."""
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be a nonempty string")
    if Path(session_id).name != session_id or session_id in (".", ".."):
        raise ValueError("session_id must be a single safe path component")
    if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    destination = Path(output_dir)
    records = []
    offset = 0.0
    for index, window in enumerate(windows):
        metrics = compute_quality_metrics(window, sample_rate)
        window_id = index
        output_path = destination / f"window_{window_id:03d}.wav"
        save_window(window, sample_rate, output_path)
        end = offset + metrics["duration_s"]
        records.append({
            "session_id": session_id,
            "window_id": window_id,
            "start_offset_s": float(offset),
            "end_offset_s": float(end),
            "duration_s": metrics["duration_s"],
            "sample_rate": int(sample_rate),
            "channels": 1,
            "status": "low_signal" if metrics["low_signal"] else "ready",
            "audio_path": str(output_path),
            "peak_amplitude": metrics["peak_amplitude"],
            "rms": metrics["rms"],
            "low_signal": metrics["low_signal"],
        })
        offset = end
    return records


def create_spark_dataframe(records, spark=None):
    """Create a Spark DataFrame with an explicit stable schema."""
    if spark is None:
        try:
            from pyspark.sql import SparkSession
        except ImportError as exc:
            raise RuntimeError("pyspark is unavailable; run this inside a Spark environment") from exc
        spark = SparkSession.builder.getOrCreate()
    from pyspark.sql.types import (
        BooleanType, DoubleType, IntegerType, StringType, StructField, StructType,
    )
    schema = StructType([
        StructField("session_id", StringType(), False),
        StructField("window_id", IntegerType(), False),
        StructField("audio_path", StringType(), False),
        StructField("start_offset_s", DoubleType(), False),
        StructField("end_offset_s", DoubleType(), False),
        StructField("duration_s", DoubleType(), False),
        StructField("sample_rate", IntegerType(), False),
        StructField("channels", IntegerType(), False),
        StructField("status", StringType(), False),
        StructField("peak_amplitude", DoubleType(), False),
        StructField("rms", DoubleType(), False),
        StructField("low_signal", BooleanType(), False),
    ])
    rows = [tuple(record[field] for field in MANIFEST_FIELDS) for record in records]
    return spark.createDataFrame(rows, schema=schema)


def write_manifest_to_delta(records, table_name="workspace.default.sleepsafe_audio_windows", spark=None):
    """Append records to a Delta table; requires an active Spark environment."""
    if not table_name or not isinstance(table_name, str):
        raise ValueError("table_name must be a nonempty string")
    frame = create_spark_dataframe(records, spark=spark)
    frame.write.format("delta").mode("append").saveAsTable(table_name)
    return len(records)
