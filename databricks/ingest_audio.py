# Databricks notebook source
# Run on Databricks compute with Unity Catalog and Auto Loader support.
# Create catalog/schema/volume and grant access before running this notebook.
from pyspark.sql import functions as F

SOURCE = "/Volumes/main/sleepsafe/audio/incoming"
CHECKPOINT = "/Volumes/main/sleepsafe/audio/checkpoints/audio_bronze"
TABLE = "main.sleepsafe.audio_bronze"

audio = (
    spark.readStream.format("cloudFiles")
    .option("cloudFiles.format", "binaryFile")
    .option("pathGlobFilter", "*.wav")
    .option("recursiveFileLookup", "true")
    .load(SOURCE)
    .withColumn("chunk_id", F.regexp_extract("path", r"([^/]+)\.wav$", 1))
    .withColumn("device_id", F.regexp_extract("path", r"_([A-Za-z0-9-]+)_[a-f0-9]+\.wav$", 1))
    .withColumn("sample_rate_hz", F.lit(16000))
    .withColumn("channels", F.lit(1))
    .withColumn("duration_seconds", F.lit(10))
    .withColumn("ingested_at", F.current_timestamp())
)

# content contains complete WAV bytes, including the header. Decode with a WAV
# reader before feeding PCM samples to a model. Keep this checkpoint stable.
query = (
    audio.writeStream.format("delta")
    .option("checkpointLocation", CHECKPOINT)
    .trigger(processingTime="10 seconds")
    .toTable(TABLE)
)
query.awaitTermination()
