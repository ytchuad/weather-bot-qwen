"""Export closed Layer A partitions to one downloadable archive."""

from __future__ import annotations

import argparse
from pathlib import Path

from layer_a.export import export_layer_a


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", dest="date_value", help="event date YYYY-MM-DD")
    parser.add_argument("--start", help="inclusive ISO timestamp or date")
    parser.add_argument("--end", help="inclusive ISO timestamp or date")
    parser.add_argument("--only-unuploaded", action="store_true")
    parser.add_argument("--verify-checksums", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = export_layer_a(
        output=args.output,
        date_value=args.date_value,
        start=args.start,
        end=args.end,
        only_unuploaded=args.only_unuploaded,
        verify_checksums=args.verify_checksums,
    )
    print(result["output"])


if __name__ == "__main__":
    main()
