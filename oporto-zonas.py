import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import *
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

data = pd.read_csv(r"C:\Users\crlss\OneDrive\Escritorio\Programming\shawking\dataset\taxiweather_data.csv")

hours = [int(datetime.fromtimestamp(i).hour) + round(float(datetime.fromtimestamp(i).minute)/60, 3) for i in data["TIMESTAMP"]]
data["HOUR"] = hours

X = [eval(i)[0] for i in data["START"]]
data["X"] = X

Y = [eval(i)[1] for i in data["START"]]
data["Y"] = Y

data.drop(["START", "Unnamed: 0.1", "Unnamed: 0", "DATE", "date_time"], axis=1, inplace=True)

def remove_outliers(df, col, threshold=3):
    z_scores = stats.zscore(df[col])
    abs_z_scores = abs(z_scores)
    return df[(abs_z_scores < threshold)]

data = remove_outliers(data, "X", 3)
data = remove_outliers(data, "Y", 7)

'''
plt.scatter(data["X"], data["Y"], alpha=0.2)
plt.show()
'''

bin_num = 6

data["X"] = pd.cut(data["X"], bin_num, labels=[i for i in range(bin_num)])
data["Y"] = pd.cut(data["Y"], bin_num, labels=[i for i in range(bin_num)])

class FeatureEncoder(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        encoder = OneHotEncoder()
        
        matrix = encoder.fit_transform(X[["WEEKDAY"]]).toarray()
        column_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        for i in range(len(matrix.T)):
            X[column_names[i]] = matrix.T[i]
        
        matrix = encoder.fit_transform(X[["X"]]).toarray()
        column_names = ["x1", "x2", "x3", "x4", "x5", "x6"]
        for i in range(len(matrix.T)):
            X[column_names[i]] = matrix.T[i]
            
        matrix = encoder.fit_transform(X[["Y"]]).toarray()
        column_names = ["y1", "y2", "y3", "y4", "y5", "y6"]
        for i in range(len(matrix.T)):
            X[column_names[i]] = matrix.T[i]
        
        return X

class FeatureDropper(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X.drop(["X", "Y", "TIMESTAMP", "WEEKDAY", "totalSnow_cm", "uvIndex.1", "moon_illumination", "moonrise", "moonset", "sunrise", "sunset", "DewPointC", "WindChillC", "pressure", "tempC", "winddirDegree"], axis=1, errors="ignore")

pipeline = Pipeline([("featureencoder", FeatureEncoder()), ("featuredropper", FeatureDropper())])
data = pipeline.fit_transform(data)

scaler = StandardScaler()

X = data.drop(["x1", "x2", "x3", "x4", "x5", "x6", "y1", "y2", "y3", "y4", "y5", "y6"], axis=1)
y = data[["x1", "x2", "x3", "x4", "x5", "x6", "y1", "y2", "y3", "y4", "y5", "y6"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(12, activation='softmax')
])

opt = Adam(learning_rate=0.0000001)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

loss, accuracy = model.evaluate(X_test, y_test)
print('Test loss:', loss)
print('Test accuracy:', accuracy)