import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


class ArrhythmiaDetector:

    def __init__(self):
        # Save accuracy
        self.train_accuracy = []
        self.test_accuracy = []
        self.highest_test_accuracy = 0
        self.hta_hl1 = 0
        self.hta_hl2 = 0

        # Save loss
        self.train_loss = []
        self.test_loss = []

        # Save node count
        self.hl1_node_count = []
        self.hl2_node_count = []

        self.reshape_divisor = 1

    # Read and preprocess the dataset
    def preprocessing(self):

        df = pd.read_csv('dataset/arrhythmia_ds.csv')

        # Replace sting values with numerical values
        df.replace('Female', 0., inplace=True)
        df.replace('Male', 1., inplace=True)

        # Convert bool values to "1"'s and "0"'s
        df = df.applymap(lambda x: 1. if x == True else x)
        df = df.applymap(lambda x: 0. if x == False else x)

        # Replace missing values in the dataset with the average value of the column it is in
        # IMPORTANT: this might introduce a bias in the network
        df = df.fillna(df.mean())

        # Remove parts of the dataset which do not vary and do not contribute any information
        df.drop(['DI: S\' wave (msec)'], 1, inplace=True)
        df.drop(['AVL: S\' wave (msec)'], 1, inplace=True)
        df.drop(['AVL: Exist ragged R wave'], 1, inplace=True)
        df.drop(['AVF: Exist ragged P wave'], 1, inplace=True)
        df.drop(['V4: Exist ragged P wave'], 1, inplace=True)
        df.drop(['V4: Exist diphasic derivation P wave'], 1, inplace=True)
        df.drop(['V5: S\' wave (msec)'], 1, inplace=True)
        df.drop(['V5: Exist ragged R wave'], 1, inplace=True)
        df.drop(['V5: Exist ragged P wave'], 1, inplace=True)
        df.drop(['V5: Exist ragged T wave'], 1, inplace=True)
        df.drop(['V6: S\' wave (msec)'], 1, inplace=True)
        df.drop(['V6: Exist diphasic derivation P wave'], 1, inplace=True)
        df.drop(['V6: Exist ragged T wave'], 1, inplace=True)
        df.drop(['DI: Amplitude S\' wave'], 1, inplace=True)
        df.drop(['AVL: Amplitude S\' wave'], 1, inplace=True)
        df.drop(['V5: Amplitude S\' wave'], 1, inplace=True)
        df.drop(['V6: Amplitude S\' wave'], 1, inplace=True)

        class_names = ["Normal",
                       "Atrial Fibrillation or Flutter",
                       "Ischemic changes (Coronary Artery Disease)",
                       "Left bundle branch block",
                       "Left ventricule hypertrophy",
                       "Old Anterior Myocardial Infarction",
                       "Old Inferior Myocardial Infarction",
                       "Others",
                       "Right bundle branch block",
                       "Sinus bradycardy",
                       "Sinus tachycardy",
                       "Supraventricular Premature Contraction",
                       "Ventricular Premature Contraction (PVC)"]

        # Replace classes with numerical values (0=normal, 1=arrhythmia)
        for c in class_names:
            if c == "Normal":
                df.replace(c, 0, inplace=True)
            else:
                df.replace(c, 1, inplace=True)

        # Convert everything to float except for the classes
        df = df.astype(float)
        df = df.astype({"Arrhythmia": int})
        print(df.head())
        return df

    # Split the dataset into training and validation
    def create_training_and_validation_data(self, df):

        # Get features
        X = df.iloc[:, :-1].values
        # Get classes
        y = df.iloc[:, -1].values

        # Split the dataset into training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Normalize features
        standard_scaler = StandardScaler()
        X_train = standard_scaler.fit_transform(X_train)
        X_test = standard_scaler.fit_transform(X_test)

        return X_train, X_test, y_train, y_test

    # Create the neural network model
    def create_model(self, hl1, hl2):

        classifier = Sequential()
        classifier.add(Dense(units=hl1, kernel_initializer='uniform', activation='relu', input_dim=262))
        classifier.add(Dense(units=hl2, kernel_initializer='uniform', activation='relu'))
        classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
        sgd = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
        classifier.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

        return classifier

    # Train model
    def train(self, hl1, hl2, X_train, X_test, y_train, y_test):

        model = self.create_model(hl1, hl2)

        # Train the model
        train = model.fit(X_train, y_train, batch_size=32, epochs=10)

        print("hl1: ", hl1, "hl2: ", hl2, train.history)
        self.train_accuracy.append(train.history['acc'][-1])
        self.train_loss.append(train.history['loss'][-1])
        self.hl1_node_count.append(hl1)
        self.hl2_node_count.append(hl2)

        # Test the model
        score = model.evaluate(X_test, y_test, verbose=0)
        self.test_loss.append(score[0])
        print('Test loss:', score[0])
        self.test_accuracy.append(score[1])
        print('Test accuracy:', score[1])

        # Save specifications for highest test accuracy
        if float(score[1]) > self.highest_test_accuracy:
            self.highest_test_accuracy = score[1]
            self.hta_hl1 = hl1
            self.hta_hl2 = hl2

    # Test training accuracy with varying number of nodes in the hidden layers
    # start_p = Minimum number of nodes per layer
    # hl1_p = Maximum number of nodes for the first hidden layer
    # hl2_p = Maximum number of nodes for the second hidden layer
    # step = Step size for node number incrementation
    def test_node_count(self, start_p, hl1_p, hl2_p, step):

        # Divisor used for plotting results
        self.reshape_divisor = int((hl1_p / step) - (start_p / step))

        X_train, X_test, y_train, y_test = self.create_training_and_validation_data(df)

        for hl1 in range(start_p, hl1_p, step):
            for hl2 in range(start_p, hl2_p, step):
                self.train(hl1, hl2, X_train, X_test, y_train, y_test)

    # Plot results in a 3d graph
    def plot_3d(self):

        x = np.reshape(self.hl2_node_count, (self.reshape_divisor, self.reshape_divisor))
        y = np.reshape(self.hl1_node_count, (self.reshape_divisor, self.reshape_divisor))
        z11 = np.reshape(self.train_accuracy, (self.reshape_divisor, self.reshape_divisor))
        z12 = np.reshape(self.train_loss, (self.reshape_divisor, self.reshape_divisor))
        z21 = np.reshape(self.test_accuracy, (self.reshape_divisor, self.reshape_divisor))
        z22 = np.reshape(self.test_loss, (self.reshape_divisor, self.reshape_divisor))

        fig1 = plt.figure()
        ax = fig1.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z11)
        ax.set_xlabel('hl1_nodes')
        ax.set_ylabel('hl2_nodes')
        ax.set_zlabel('train_accuracy')

        fig2 = plt.figure()
        ax = fig2.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z12)
        ax.set_xlabel('hl1_nodes')
        ax.set_ylabel('hl2_nodes')
        ax.set_zlabel('train_loss')

        fig3 = plt.figure()
        ax = fig3.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z21)
        ax.set_xlabel('hl1_nodes')
        ax.set_ylabel('hl2_nodes')
        ax.set_zlabel('test_accuracy')

        fig4 = plt.figure()
        ax = fig4.add_subplot(111, projection='3d')
        ax.plot_surface(x, y, z22)
        ax.set_xlabel('hl1_nodes')
        ax.set_ylabel('hl2_nodes')
        ax.set_zlabel('test_loss')

        plt.show()


if __name__ == '__main__':
    detector = ArrhythmiaDetector()
    df = detector.preprocessing()
    detector.test_node_count(100, 1000, 1000, 100)
    print('Highest test accuracy: {:.3%} with hl1:{} hl2:{}'.format(detector.highest_test_accuracy,
                                                                    detector.hta_hl1,
                                                                    detector.hta_hl2))
    detector.plot_3d()