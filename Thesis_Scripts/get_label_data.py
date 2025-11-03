import calendar
import sys

import pandas as pd
from matplotlib import pyplot as plt


def plot_month_histogram(df):
    dt = pd.to_datetime(
        df["Recording Begin Time"],
        format="%d.%m.%Y %H:%M:%S",
        dayfirst=True)

    df = df.assign(_month=dt.dt.month)

    df = df.dropna(subset=["_month"])

    dedup = df[["Location", "_month"]].drop_duplicates()
    month_counts = (
        dedup["_month"]
        .value_counts()
        .reindex(range(1, 13), fill_value=0)
        .sort_index()
    )

    print(month_counts)

    plt.figure(figsize=(10, 5))
    plt.bar(month_counts.index,
            month_counts.values,
            tick_label=[calendar.month_abbr[m] for m in month_counts.index])

    plt.xlabel("Month")
    plt.ylabel("Count")
    plt.title("Histogram of months")
    plt.tight_layout()
    plt.show()


def main():
    path = "C:\\Users\\ander\\Github\\Masters\\Pdata\\labels.xlsx"
    df = pd.read_excel(path)
    plot_month_histogram(df)


if __name__ == '__main__':
    main()
