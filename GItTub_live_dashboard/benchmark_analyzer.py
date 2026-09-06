"""
Run this AFTER you've done some test sessions, to get a report:
    python benchmark_analyzer.py
"""
import pandas as pd


def compute_metrics(csv_path="logs/session_logs.csv"):
    df = pd.read_csv(csv_path)
    if df.empty:
        print("No log data available yet. Run some sessions first.")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    total_hours = (df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 3600.0
    total_hours = max(total_hours, 1.0 / 60.0)  # avoid divide-by-zero on short tests

    total_connections = len(df)
    true_positives = len(df[df["verification_result"] == "TRUE_POSITIVE"])
    false_positives = len(df[df["verification_result"] == "FALSE_POSITIVE"])
    fp_per_hour = false_positives / total_hours
    avg_duration = df[df["verification_result"] == "TRUE_POSITIVE"]["duration_ms"].mean()

    print("=" * 45)
    print(" GITTUB SYSTEM BENCHMARK REPORT ")
    print("=" * 45)
    print(f"Total Test Uptime       : {total_hours:.2f} Hours")
    print(f"Total Trigger Events    : {total_connections}")
    print(f"Verified Triggers (TP)  : {true_positives}")
    print(f"False Triggers (FP)     : {false_positives}")
    print(f"False Positives / Hour  : {fp_per_hour:.2f} FP/hr")
    print(f"Avg Stream Duration     : {avg_duration:.1f} ms")
    print("=" * 45)


if __name__ == "__main__":
    compute_metrics()
