import polars as pl

from polars_profiling import ProfileReport
from polars_profiling.utils.cache import cache_file

if __name__ == "__main__":
    file_name = cache_file(
        "titanic.csv",
        "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv",
    )

    df = pl.read_csv(file_name)

    profile = ProfileReport(df, title="Titanic Dataset", explorative=True)
    profile.to_file("titanic_report.html")
