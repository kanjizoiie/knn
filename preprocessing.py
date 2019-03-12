import numpy as np
import pandas

def symmetric_uncertainty(A, B):
    mean_a = np.mean(A)
    mean_b = np.mean(B)
    upper = np.sum([a - mean_a for a in A], dtype=float) * np.sum([b - mean_b for b in B], dtype=float)
    lower = np.sqrt(np.sum([(a - mean_a)**2 for a in A], dtype=float) * np.sum([(b - mean_b)**2 for b in B], dtype=float), dtype=float)
    return (upper / lower)

results = []

# Load the data from the CSV file
data_frame = pandas.read_csv("spambase/spambase.csv")
# Randomize the dataframe
data_frame = data_frame.sample(frac=1).reset_index(drop=True)

# Calculate symmetric uncertainty for the whole class.
for counter, elem in enumerate(data_frame.iteritems()):
    for c in range(counter + 1, len(data_frame.columns)):
        res = symmetric_uncertainty(elem[1], data_frame.iloc[:, c])
        results.append([(elem[0] + "|" + data_frame.iloc[:, c].name), res])

np.set_printoptions(threshold=np.inf)
print(np.asarray(results))