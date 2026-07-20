#!/usr/bin/env python3
"""CLI for analytics.db — same numbers as the bot's /stats, from the terminal.

    ./stats.py            # last 30 days
    ./stats.py 7           # last 7 days
"""
import sys

import analytics

if __name__ == "__main__":
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    print(analytics.format_summary(analytics.summary(days)))
