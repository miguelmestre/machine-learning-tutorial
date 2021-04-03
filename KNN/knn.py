import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing
import pandas as pd
import numpy as np

data = pd.read_csv("car.data")


# Transform alfa values on integer values
encoder = preprocessing.LabelEncoder()
buying = encoder.fit_transform(list(data["buying"]))
maint = encoder.fit_transform(list(data["maint"]))
doors = encoder.fit_transform(list(data["doors"]))
persons = encoder.fit_transform(list(data["persons"]))
lug_boost = encoder.fit_transform(list(data["lug_boost"]))
safety = encoder.fit_transform(list(data["safety"]))
cls = encoder.fit_transform(list(data["class"]))

predict = "class"

X = list(zip(buying, maint, doors, persons, lug_boost, safety))
Y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)

model= KNeighborsClassifier(n_neighbors=5 )

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(acc)


predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]
for x in range(len(predicted)):
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])
