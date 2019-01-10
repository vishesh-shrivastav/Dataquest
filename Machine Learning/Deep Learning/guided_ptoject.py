from sklearn.datasets import load_digits
import pandas as pd
import matplotlib.pyplot as plt
# load_digits() is a dictionary. The image is 
# contained in the key "data"
df = pd.DataFrame(load_digits()["data"])
df.shape

# Plotting first image
%matplotlib inline
# Plot first image
first_image = df.iloc[0]
np_image = first_image.values
# Reshape to 8*8
np_image = np_image.reshape(8,8)
plt.imshow(np_image, cmap="gray_r")

# Plotting multiple images
w = 10
h = 10
fig=plt.figure(figsize=(8, 8))
columns = 4
rows = 2
indexes = [0,100,200,300,1000,1100,1200,1300]

for i in range(1, columns*rows+1):
    img = df.iloc[indexes[i-1]].values.reshape(8,8)
    fig.add_subplot(rows, columns, i)
    plt.imshow(img, cmap="gray_r")
plt.show()

# Targets
target = pd.DataFrame(load_digits()["target"])
target.shape


# Setup arrays to store train and test accuracies
neighbors = np.arange(1, 9)
train_accuracy = np.empty(len(neighbors))
test_accuracy = np.empty(len(neighbors))

# Loop over different values of k
for i, k in enumerate(neighbors):
    # Setup a k-NN Classifier with k neighbors: knn
    knn = KNeighborsClassifier(n_neighbors=k)

    # Fit the classifier to the training data
    knn.fit(X_train, y_train)
    
    #Compute accuracy on the training set
    train_accuracy[i] = knn.score(X_train, y_train)

    #Compute accuracy on the testing set
    test_accuracy[i] = knn.score(X_test, y_test)

# Generate plot
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracy, label = 'Testing Accuracy')
plt.plot(neighbors, train_accuracy, label = 'Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

################################################################
from sklearn.model_selection import cross_val_score, KFold

def train(k):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    return knn

def test():
    knn.score(X_test, y_test)
    return

n_neighbors = [1,2,3,4,5,6,7,8,9]
train_accuracies = []
def cross_validate():
	kf = KFold(4, shuffle=True, random_state=1)
	for i in n_neighbors:
		model = train(i)
		train_acc = cross_val_score(model, X_train, X_test, scoring="accuracy", cv = kf)
		train_accuracies.append(train_acc)
    return train_accuracies

####################################################################
from sklearn.model_selection import cross_val_score, KFold
kf = KFold(5, shuffle=True, random_state=1)
model = KNeighborsRegressor()
mses = cross_val_score(model, dc_listings[["accommodates"]], dc_listings["price"], scoring="neg_mean_squared_error", cv=kf)
rmses = np.sqrt(np.absolute(mses))
avg_rmse = np.mean(rmses)

print(rmses)
print(avg_rmse)

#####################################################################
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(df, targets, test_size = 0.2, 
                                                    random_state=42, stratify=targets)

# Create a k-NN classifier with 7 neighbors: knn
knn = KNeighborsClassifier(n_neighbors = 7)

# Fit the classifier to the training data
knn.fit(X_train,y_train)

# Print the accuracy
print(knn.score(X_test, y_test))



#########################################################################
def train(k):
    knn = KNeighborsClassifier(n_neighbors=k)
    #knn.fit(X_train, y_train)
    return knn

def test():
    knn.score(X_test, y_test)
    
n_neighbors = [1,2,3,4,5,6,7,8,9]
    
def cross_validate():
    train_accuracies = []
    test_accuracies = []
    kf = KFold(4, shuffle=True, random_state=1)
    for i in n_neighbors:
        model = train(i)
        train_acc = cross_val_score(model, X_train, y_train, 
                                    scoring="accuracy", cv = kf)
        train_accuracies.append(train_acc)
        
        test_acc = cross_val_score(model, X_test, y_test, 
                                    scoring="accuracy", cv = kf)
        test_accuracies.append(test_acc)
    print(test_accuracies)
    return train_accuracies
from sklearn.model_selection import cross_val_score, KFold
cross_validate()
