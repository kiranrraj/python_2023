import pickle, os
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

ROOT_DIR = r'D:\github\python_2023\project_experiments'
DATA_FILE = os.path.join(ROOT_DIR, "data.pickle")

data_dict = pickle.load(open(DATA_FILE, 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])
# print(labels)
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()
model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print("Accuray: {}".format(score * 100))

location = os.path.join(ROOT_DIR, 'model.p')
print(location)
f = open(location, 'wb')
pickle.dump({'model':model}, f)
f.close()