"""
This script is a demonstration of using the K-Nearest Neighbors
algorithm to classify types of glass in a crime scene. 

Dataset Creator: B. German, Central Research Establishment, Home Office
Forensic Science Service, Aldermaston, Reading, Berkshire RG7 4PN

Dataset Donor: Vina Spiehler, Ph.D., DABFT, Diagnostic Products Corporation

More Info: http://ugrad.stat.ubc.ca/R/library/e1071/html/Glass.html
"""

## The study of classification of types of glass was motivated by
## criminological investigation. At the scene of the crime, the glass
## left can be used as evidence...if it is correctly identified!

## Attribute Information:
##    1. Id number: 1 to 214
##    2. RI: refractive index
##    3. Na: Sodium (unit measurement: weight percent in corresponding oxide, as
##                                     are attributes 4-10)
##    4. Mg: Magnesium
##    5. Al: Aluminum
##    6. Si: Silicon
##    7. K: Potassium
##    8. Ca: Calcium
##    9. Ba: Barium
##   10. Fe: Iron
##   11. Type of glass: (class attribute)
##       -- 1 building_windows_float_processed
##       -- 2 building_windows_non_float_processed
##       -- 3 vehicle_windows_float_processed
##       -- 4 vehicle_windows_non_float_processed (none in this database)
##       -- 5 containers
##       -- 6 tableware
##       -- 7 headlamps

from sklearn.neighbors import KNeighborsClassifier
import numpy
import matplotlib.pyplot as plt

# prettier printing
numpy.set_printoptions(precision=6, suppress=True)

# this is the data we are going to use
glass = numpy.genfromtxt('glass.csv', delimiter=',')

# dimensions of the array
print glass.shape
n = glass.shape[0]
p = glass.shape[1] - 1

# the first 7 rows/observations
print glass[:7,:]

# these are the 7 classes.
glass_types = ["bui_f","bui_nf","veh_f","container","tableware","headlamp"]

# define the training set to be a sample 
train_size = 120
train_cases = numpy.random.choice(n, train_size)

# the remaining observations will be the test cases
test_cases = numpy.array([x for x in range(n) if x not in train_cases])

train_data = glass[train_cases,1:9]
train_labels = glass[train_cases, 10]

test_data = glass[test_cases, 1:9]

# the test_labels are the classes we want to get right with knn. these
# are not part of the "dataset"
test_labels = glass[test_cases, 10] 

# the unique test labels
set(test_labels)

# just taking a look at the data
print train_data
print train_labels
print test_data

#######################################

## 1-NN

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(train_data, train_labels)

knn_predictions = knn.predict(test_data)

# multiply by 1.0 to make test_error a float
test_error = (test_labels != knn_predictions).sum() * 1.0 / len(test_cases)

### Confusion Matrix
cm = confusion_matrix(test_labels, knn_predictions)
print cm

## Show confusion matrix in nicer way, separate window

# clear current figure, just in case.
plt.clf()

plt.matshow(cm)
plt.title('Confusion matrix')
plt.ylabel('True labels')
plt.xlabel('KNN Predicted labels')
plt.colorbar()

# print the actual error in the colored boxes
for i, row in enumerate(cm):
    for j, error in enumerate(row):
        if error > 0:
            plt.text(j-0.2, i+0.2, error, fontsize=14)

# x and y axes labels
plt.xticks(range(len(glass_types)), glass_types)
plt.yticks(range(len(glass_types)), glass_types)

# might need to make image a bit larger to show everything
plt.show()

###############

## KNN with varying K
Kmax = 30
test_error = [0]

for k in range(1, Kmax + 1):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(train_data, train_labels)
    knn_predictions = knn.predict(test_data)
    error = (test_labels != knn_predictions).sum() * 1.0 / len(test_cases)
    test_error.append(error)

# plot the results
plt.clf()
plt.plot(test_error)
plt.ylabel('Error')
plt.xlabel('Chosen "K"')
plt.xlim(1, Kmax)
plt.show()

## TODO: maybe Leave-one-out cross validation 

    
