import os
from datetime import datetime

REPORT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'output', 'pipeline_report.txt')


def log_step(step: str, summary: str):
    """
    Appends a single summary line for a pipeline step to the shared report file.

    Args:
        step   : Step identifier, e.g. 'Step 2 - Cleaning'
        summary: One-line summary of the step result, e.g. '1,500 → 1,097 rows retained (73.1%)'

    Example output in pipeline_report.txt:
        [2026-04-17 10:23:01] Step 2 - Cleaning         : 1,500 → 1,097 rows retained (73.1%)
    """
    os.makedirs(os.path.dirname(REPORT_FILE), exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{timestamp}] {step:<35}: {summary}\n"
    with open(REPORT_FILE, 'a', encoding='utf-8') as f:
        f.write(line)
