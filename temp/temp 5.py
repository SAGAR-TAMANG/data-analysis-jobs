import pandas as pd

# Load the DataFrame containing the rows to compare against
df = pd.read_excel('NaukriJobListing_2023-07-14.xlsx')
# Define the text to compare
text_to_compare = "analyst"

is_present = df.isin([text_to_compare])
indices = is_present[is_present.any(axis=1)].index.tolist()

if len(indices) > 0:
    print("The text is present at the following indices:")
    for idx in indices:
        print(idx)
else:
    print("The text is not present in any row.")
