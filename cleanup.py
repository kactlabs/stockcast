"""Remove generated stock analysis report .md files and clean index.md links."""

import glob
import os
import re


def cleanup():
    # Delete all generated report files
    reports = glob.glob("stock_analysis_report_*.md")
    for f in reports:
        os.remove(f)
        print(f"Deleted: {f}")

    # Clean index.md — remove lines that link to report files
    index_path = "index.md"
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as fh:
            lines = fh.readlines()
        cleaned = [l for l in lines if not re.search(r"stock_analysis_report_\d+_\d+\.md", l)]
        with open(index_path, "w", encoding="utf-8") as fh:
            fh.writelines(cleaned)
        print("Cleaned index.md")

    print(f"Done. Removed {len(reports)} report(s).")


if __name__ == "__main__":
    cleanup()
