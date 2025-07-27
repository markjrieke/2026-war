from polars import DataFrame, from_pandas
from xarray import DataArray

def from_xarray(data: DataArray) -> DataFrame:
    return data.to_dataframe().pipe(from_pandas, include_index=True)
