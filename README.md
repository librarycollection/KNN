#K-Nearest Neighbors Example  

##Load training and test data  
We are loading two CSV files, each file has a set of points - one file has already been classified while the other has not.
We load the input data in a Python dictionary so each (x,y) tuple corresponds to a classification. The unclassified data is entered as a list of tuples for the time being.  

```
import csv
import sys
x = []
y = []
z = []
with open('input_data.csv','rb') as f:
    reader = csv.reader(f)
    for row in reader:
        x.append(float(row[0]))
        y.append(float(row[1]))
        z.append(row[2])
        
coordinates = zip(x,y)
input_data = {coordinates[i]:z[i] for i in xrange(len(coordinates))}

test_x = []
test_y = []
with open('test_data.csv', 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        test_x.append(float(row[0]))
        test_y.append(float(row[1]))
        
test_coordinates = zip(test_x,test_y)
```
##Distance  
The KNN algorithm classifies an item based on the K closest points. We need to define "closest" - in this context, Euclidean distance makes sense since the points are on a plane. Other contexts, however, may require different definitions of distance (e.g. Cosine similarity). Euclidean distance can be represented by the square root of the sum of squared distances between two vectors.  

```
def euclidean_distance(x, y):
    if len(x) != len(y):
        return "Error: try equal length vectors"
    else:
        return sqrt(sum([(x[i]-y[i])**2 for i in xrange(len(y))]))
```

##Neighbors  
The next step involves writing a function to determine the K nearest neighbors to a given point. To do so, we calculate the pairwise distances between the given point and the set of trained points. We then sort the results and slice off the first K elements.

```
def neighbors(k, trained_points, new_point):
    neighbor_distances = {}
    
    for point in trained_points:
        if point not in neighbor_distances:
            neighbor_distances[point] = euclidean_distance(point, new_point)
    
    least_common = sorted(neighbor_distances.items(), key = lambda x: x[1])
    
    k_nearest_neighbors = zip(*least_common[:k])
    
    return list(k_nearest_neighbors[0])
```

##Classifier  
With the K-nearest neighbors, we now just need to determine the appropriate classification for our point. Python's collection module has a Counter object that works well for this purpose. The Counter object counts number of classifications among the neighbors and we assign the most common classification to the data point. If there is a tie among classifications, the first to appear in the Counter is chosen. Potential solutions include altering the code to randomly choose among the tied items or choosing a different K.

```
from collections import Counter
def knn_classifier(neighbors, input_data):
    knn = [input_data[i] for i in neighbors]
    knn = Counter(knn)
    classifier, _ = knn.most_common(1)[0]
    
    return classifier
```

##Putting it Together

```
results = {}
for item in test_coordinates:
    results[item] = knn_classifier(neighbors(3,input_data.keys(), item), input_data)
```