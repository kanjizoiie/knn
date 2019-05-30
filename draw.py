import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import pandas
from sklearn.utils import shuffle

# Load the data from the CSV file
data_frame = pandas.read_csv("spambase/spambase.csv")
# Randomize the dataframe
data_frame = shuffle(data_frame)
sample_frame = data_frame.sample(200)
sample_frame_words = sample_frame.iloc[:, ::3]
print(sample_frame_words)

scatter_matrix(sample_frame_words, alpha=0.2)
plt.show()