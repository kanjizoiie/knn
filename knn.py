import numpy as np
from collections import Counter
import time
from multiprocessing import Pool
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import pandas
from sklearn.utils import shuffle


# SETTINGS
number_of_cross_validations = 10


# Calculates the euclidian distance between two N-dimensional points
def euclideanDistance(A, B):
    return np.sqrt(np.sum([np.square(B[i] - A[i]) for i in range(len(A))]))

# Class that defines a KNN classifier
class knn:
    def __init__(self, x, target):
        self.pool = Pool(4)
        if (len(x) != len(target)):
            raise Exception("Length of data and target must be the same")
        else:
            self.x = x
            self.target = target

    # Predict the classes for an array of tuples containing the data
    def predict(self, query, k):
        predictions = []
        predictions_time = []
        for elem in query:
            # Start timing for evaluation
            start = time.process_time_ns()
            # If there is a length mismatch raise error
            if len(elem) != len(self.x[0]):
                raise Exception("Query and data length mismatch")
            distance_from_query_elem = []
            k = int(k)

            query_elem_arr = [elem for i in range(len(self.x))]
            e_start = time.process_time_ns()
            # Calculate the distances from the query point to all other points in the array
            distances = self.pool.starmap(euclideanDistance, zip(self.x, query_elem_arr))
            e_end = time.process_time_ns()
            for index, value in enumerate(distances):
                distance_from_query_elem.append((value, index))
            # Sort the distances from the query
            sorted_distance_from_query_elem = sorted(distance_from_query_elem)

            # Choose the k nearest values
            k_nearest_distances_indicies = sorted_distance_from_query_elem[1:k + 1]
            # Get the labels from the k nearest neighbors.
            k_nearest_labels = [self.target[i] for distance, i in k_nearest_distances_indicies]

            # Count the labels in the k nearest labels
            label_count = Counter(k_nearest_labels)
            # Stop timing for evaluation
            end = time.process_time_ns()
            # Add the prediction to the predictions
            predictions.append((max(label_count, key=label_count.get), label_count))
            predictions_time.append(((end - start), (e_end - e_start)))
        # Return the predictions
        return predictions, predictions_time

accs = []
predictions = []

if __name__ == '__main__':
    # Load the data from the CSV file
    data_frame = pandas.read_csv("spambase/spambase.csv")
    # Randomize the dataframe
    data_frame = shuffle(data_frame)
    # Split data into a number of chunks
    df_split = np.array_split(data_frame, number_of_cross_validations)

    for i in range(len(df_split)):
        print("Running: Split", i)
        copied_arr = df_split.copy()
        test = copied_arr.pop(i)
        train = pandas.concat(copied_arr)

        # Split the data into data and class
        train_data, train_target = train.iloc[:, :-1], train.iloc[:, -1]
        test_data, test_target = test.iloc[:, :-1], test.iloc[:, -1]

        # Print the shapes of the dataframes
        print("Train:", train_data.shape, " Test:", test_data.shape)

        # Send the training dataframe values into the KNN classifier
        k = knn(train_data.values, train_target.values)

        # Make a prediction on the testing data
        prediction = k.predict(test_data.values, np.sqrt(train_data.shape[0]))
        print(prediction)
        # Calculate how many errors in prediction
        error_prediction = 0

        true_positive = 0
        false_positive = 0
        false_negative = 0
        true_negative = 0


        for index, elem in enumerate(test_target.values):
            if elem != prediction[0][i][0]:
                error_prediction += 1
            if bool(elem) == True and bool(prediction[0][i][0]) == True:
                true_positive += 1
            elif bool(elem) == False and bool(prediction[0][i][0]) == True:
                false_positive += 1
            elif bool(elem) == False and bool(prediction[0][i][0]) == False:
                false_negative += 1
            elif bool(elem) == True and bool(prediction[0][i][0]) == False:
                true_negative += 1
        # Print the amount of errors in the prediction
        print("Length of test data: ", test_data.shape[0], " Errors in prediction: ", error_prediction)
        print("Accuracy: ", (test_data.shape[0] - error_prediction) / test_data.shape[0])
        print("True positive:", true_positive)
        print("False positive:", false_positive)
        print("False negative:", false_negative)
        print("True negative:", true_negative)
        accs.append((test_data.shape[0] - error_prediction) / test_data.shape[0])
    print(accs)
    print("Average:", np.average(accs))
