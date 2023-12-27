# cd into lat-lon-data directory before running

import pandas as pd

df = pd.read_csv("lat-lon-data/airport-codes.csv")
df = df[df.iso_country == "US"]

# df.column_name != whole string from the cell
# now, all the rows with the column: Name and Value: "dog" will be deleted

df.to_csv("lat-lon-data/us-airport-codes.csv", index=False)
