import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix, radviz
import pandas
from sklearn.utils import shuffle

# Load the data from the CSV file
data_frame = pandas.read_csv("spambase/spambase.csv")
# Randomize the dataframe
data_frame = shuffle(data_frame)
sample_frame = data_frame.sample(frac=1)
sample_frame = sample_frame.iloc[:, ::1]

print(sample_frame)
radviz(sample_frame, "spam")
plt.show()

# scatter_matrix(sample_frame, alpha=0.2)
# plt.show()