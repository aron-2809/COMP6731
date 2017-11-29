import pandas as pd
import numpy as np
from time import time
import os
from sklearn.model_selection import train_test_split, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler  # scipy package
from sklearn.metrics import accuracy_score
from sklearn import cross_validation, neighbors
from sklearn import svm
import pickle

class IntrusionDetector:

    def __init__(self, data_path):
        self.kdd_path = data_path
        self.kdd_data = []
        self.kdd_data_processed = []
        self.kdd_data_reduced = []
        self.kdd_train_data = []
        self.kdd_test_data = []
        self.get_data()

    def get_data(self):
        col_names = ["duration","protocol_type","service","flag","src_bytes",
            "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
            "logged_in","num_compromised","root_shell","su_attempted","num_root",
            "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
            "is_host_login","is_guest_login","count","srv_count","serror_rate",
            "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
            "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
            "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
            "dst_host_rerror_rate","dst_host_srv_rerror_rate","label"]
        self.kdd_data = pd.read_csv(self.kdd_path, header=None, names = col_names)
        # print(kdd_data.describe())
        #print (kdd_data.shape)

    # To reduce labels into "Normal" and "Attack"
    def get_2classes_labels(self):
        label_2class = self.kdd_data['label'].copy()
        label_2class[label_2class != 'normal.'] = 'attack.'
        label_2class = label_2class.values.reshape((label_2class.shape[0], 1))
        return label_2class

    def preprocessor(self):
        nominal_features = ["protocol_type", "service", "flag"]  # [1, 2, 3]
        binary_features = ["dst_bytes", "num_failed_logins", "num_compromised",\
                           "root_shell", "num_outbound_cmds", "is_host_login"]  # [6, 11, 13, 14, 20, 21]
        numeric_features = [
            "duration", "src_bytes",
            "land", "wrong_fragment", "urgent", "hot",
            "logged_in", "su_attempted", "num_root",
            "num_file_creations", "num_shells", "num_access_files",
            "is_guest_login", "count", "srv_count", "serror_rate",
            "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
            "diff_srv_rate", "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
            "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
            "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
            "dst_host_rerror_rate", "dst_host_srv_rerror_rate"
        ]

        #convert nominal features to numeric features
        #nominal features: ["protocol_type", "service", "flag"]
        kdd_nominal = self.kdd_data[nominal_features].stack().astype('category').unstack()

        kdd_nominal_encoded = np.column_stack((kdd_nominal["protocol_type"].cat.codes,\
                                               kdd_nominal["service"].cat.codes,\
                                               kdd_nominal["flag"].cat.codes))
        #print (kdd_nominal_encoded)
        #kdd_nominal_encoded = pd.get_dummies(kdd_nominal, columns=nominal_features)

        kdd_binary = self.kdd_data[binary_features]

        # Standardizing and scaling numeric features
        kdd_num = self.kdd_data[numeric_features].astype(float)
        kdd_num_std = StandardScaler().fit_transform(kdd_num)

        #TO-DO: kdd_nominal_encoded is ignored because of memory error when fitting classifier
        self.kdd_data_processed = np.concatenate([kdd_num_std, kdd_binary , kdd_nominal_encoded], axis=1)
        #kdd_data_processed = np.concatenate([kdd_data_processed, kdd_nominal_encoded], axis = 1)
        #print(kdd_data_processed.shape)

    def feature_reduction_PCA(self):
        kdd_num = self.kdd_data_processed[:,:-9]
        kdd_binary = self.kdd_data_processed[:, -9:-3]
        kdd_nominal_encoded = self.kdd_data_processed[:, -3:]
        #print (kdd_processed.shape)
        #print (kdd_binary.shape)

        #compute Eigenvectors and Eigenvalues
        mean_vec = np.mean(kdd_num, axis=0)
        cov_mat = np.cov((kdd_num.T))

        # Correlation matrix
        cor_mat = np.corrcoef((kdd_num.T))
        eig_vals, eig_vecs = np.linalg.eig(cor_mat)
        #print ('\n\n eigenvectors \n %s' %eig_vecs)
        #print ('\n\n eigenvalues \n %s' %eig_vals)

        # To check that the length of eig_vectors is 1
        for ev in eig_vecs:
            np.testing.assert_array_almost_equal(1.0,np.linalg.norm(ev))
        #print ('Everything ok!')

        #to rank the eigenvalues from highest to lowest in order choose the top k eigenvectors and ignore the rest
        # 1- Make a list of (eigenvalue, eigenvector) tuples
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

        # 2- Sort the (eigenvalue, eigenvector) tuples from high to low
        eig_pairs.sort()
        eig_pairs.reverse()

        # 3- Visually confirm that the list is correctly sorted by decreasing eigenvalues
        #print('\n\nEigenvalues in descending order:')
        #for i in eig_pairs:
        #    print(i[0])

        #feature reduction
        # just the 10 first are greater 1 and one which is close to 1 => pick 11
        matrix_w = np.hstack((eig_pairs[0][1].reshape(32,1),
                              eig_pairs[1][1].reshape(32,1),
                              eig_pairs[2][1].reshape(32,1),
                              eig_pairs[3][1].reshape(32,1),
                              eig_pairs[4][1].reshape(32,1),
                              eig_pairs[5][1].reshape(32,1),
                              eig_pairs[6][1].reshape(32,1),
                              eig_pairs[7][1].reshape(32,1),
                              eig_pairs[8][1].reshape(32,1),
                              eig_pairs[9][1].reshape(32,1),
                              eig_pairs[10][1].reshape(32,1)))
        #print('Matrix W:\n', matrix_w)
        '''
        from sklearn.decomposition import PCA as sklearnPCA
        sklearn_pca = sklearnPCA(n_components=10)
        kdd_num_projected = sklearn_pca.fit_transform(kdd_num_std)
        #print (kdd_num_projected)
        '''

        #projection to new feature space
        kdd_num_projected = kdd_num.dot(matrix_w)
        #print (kdd_num_projected.shape)
        self.kdd_data_reduced = np.concatenate([kdd_num_projected, kdd_binary, kdd_nominal_encoded], axis=1)
        #print (self.kdd_data_reduced.shape)

    def classify(self):
        # add label to the finalized data set
        self.kdd_data_reduced = np.concatenate([self.kdd_data_reduced, self.get_2classes_labels()], axis=1)
        # split data to 80% for train and 20% for test
        # shuffle is by default = True
        self.kdd_train_data, self.kdd_test_data = train_test_split(self.kdd_data_reduced, train_size=0.8)

        # TO-DO: Split kdd_train_data for cross-validation
        # classifier
        # Create a Gaussian Classifier
        model = GaussianNB()
        # Train the model using the training sets
        model.fit(self.kdd_train_data[:, :-1], self.kdd_train_data[:, -1])
        with open('naivebayes.pickle','wb') as f:
            pickle.dump(model,f)

        # Predict
        predicts = model.predict(self.kdd_test_data[:, :-1])
        return (accuracy_score(self.kdd_test_data[:, -1], predicts))
        '''# Create SVM classification object
        model = svm.SVC(kernel='linear', C=1, gamma=1)
        # there is various option associated with it, like changing kernel, gamma and C value. Will discuss more # about it in next section.Train the model using the training sets and check score
        model.fit(self.kdd_train_data[:, :-1], self.kdd_train_data[:, -1])
        model.score(self.kdd_train_data[:, :-1], self.kdd_train_data[:, -1])
        # Predict Output
        predicts = model.predict(self.kdd_test_data[:, :-1])
        return (accuracy_score(self.kdd_test_data[:, -1], predicts))
        '''

    def knearestneighbor(self):
        print('in knn')
        self.kdd_data_reduced=np.concatenate([self.kdd_data_reduced, self.get_2classes_labels()], axis=1)
        self.kdd_train_data, self.kdd_test_data = train_test_split(self.kdd_data_reduced, test_size=0.2)
        clf=neighbors.KNeighborsClassifier()
        clf.fit(self.kdd_train_data)
        print('model trained')
        predicts = clf.predict(self.kdd_test_data[:, :-1])
        accuracy=accuracy_score(self.kdd_test_data[:, -1], predicts)
        print('knn accuracy',accuracy)

    # def post_classification(self):
    #     #print(len(self.kdd_train_data),len(self.kdd_test_data))
    #     # X_train, X_test, y_train, y_test =cross_validation.train_test_split(self.kdd_train_data,self.kdd_test_data,test_size=0.2)
    #     # #self.model.fit(self.kdd_train_data,self.kdd_test_data)
    #     # accuracy=self.model.score(X_test,y_test)
    #     # print("Post Classification: "+accuracy)





def main():
    # Data path
    cwd = os.getcwd()  # current directory path
    kdd_data_path = cwd + "/data/kddcup.data_10_percent/kddcup.data_10_percent_corrected"

    i_detector = IntrusionDetector(kdd_data_path)
    i_detector.preprocessor()

    i_detector.feature_reduction_PCA()
    i_detector.knearestneighbor()
    # accuracy = i_detector.classify()
    # print("accuracy of classifying 20% test-data is : ")
    # print(accuracy)
    #i_detector.post_classification()

if __name__ == '__main__':
    main()
