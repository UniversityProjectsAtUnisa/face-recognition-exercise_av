import os
import pickle as pk
import numpy as np
import cv2
from face_aligner.FaceOperator import FaceOperator
from filters.SharpeningFilter import SharpeningFilter
from PCA.pca_sklearn import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm


from utils import (
    get_dataset_filelist,
    load_features,
    Labeler
)

MODELS_DIR = "models"
FEATURES_MODEL = 'features_model.pkl'
FACE_RECOGNITION_MODEL = 'predictor.pkl'
LABELS_FILENAME = 'labels.pkl'

_sharpeningFilter = SharpeningFilter(th1=100, th2=200)
_faceOperator = FaceOperator(models_dir=MODELS_DIR, box_padding=10)
_clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))


class FaceRecognition:

    def __init__(self):
        '''FaceRecognition model constructor
        '''
        tuned_parameters = [
            {"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000, 1e9], "shrinking": [True, False]},
            {"kernel": ["linear"], "C": [0.001, 0.01, 0.1, 1, 10, 100], "shrinking": [True, False]},
        ]
        self.model = GridSearchCV(SVC(probability=True), tuned_parameters, verbose=2, n_jobs=-1)
        self.rejection_treshold = 0.0

    def fit(self, X_train, y_train, X_val, y_val, tuning_rounds=10):
        '''FaceRecognition model training. The features have been already extracted.
        '''
        print("Fit")
        self.model.fit(X_train, y_train)
        print(self.model.best_estimator_)
        print("Tuning")
        # Tune the rejection threshold on the validation set
        self.tune_rejection_threshold(X_val, y_val, tuning_rounds)

    def tune_rejection_threshold(self, X_val, y_val, rounds=10):
        '''Tuning of the rejection threshold.
        '''
        left = 0.0
        right = 1.0
        best_rt = 1.0
        best_acc = 0.0
        for _ in tqdm(range(rounds), desc="Optimizing rejection threshold"):
            self.rejection_treshold = left
            results = self.predict(X_val)
            left_acc = accuracy_score(y_val, results)

            self.rejection_treshold = right
            results = self.predict(X_val)
            right_acc = accuracy_score(y_val, results)

            if left_acc >= right_acc:
                if left_acc >= best_acc:
                    best_rt = left
                    best_acc = left_acc
                right = (right+left)/2
            else:  # right è il migliore
                if right_acc > best_acc:
                    best_rt = right
                    best_acc = right_acc
                left = (left+right)/2

        self.rejection_treshold = best_rt


    def predict(self, X):
        '''Predicts the identities of a list of faces. The features have been already extracted.
           The label is a number between 0 and N-1 if the face is recognized else -1.
           X is a list of examples to predict.
        '''
        probs = self.model.predict_proba(X)
        classes = np.argmax(probs, axis=1)
        results = []
        for i, c in enumerate(classes):
            if probs[i, c] < self.rejection_treshold:
                results.append(-1)
            else:
                results.append(c)
        return np.array(results)

    def save(self, output_path=os.path.join(MODELS_DIR, FACE_RECOGNITION_MODEL)):
        '''Saves model to be delivered in the pickle format.

           with open(file_path,'wb') as file:
               pk.dump(self.model, file)
        '''
        with open(output_path, 'wb') as file:
            pk.dump(dict(model=self.model, th=self.rejection_treshold), file)

    def load(self, input_path=os.path.join(MODELS_DIR, FACE_RECOGNITION_MODEL)):
        '''Loads the model from a pickle file.

           with open(file_path,'rb') as file:
               self.model = pk.load(file)
        '''
        with open(input_path, 'rb') as file:
            data = pk.load(file)
            self.model = data["model"]
            self.rejection_treshold = data["th"]


def preprocessing(bgr_image):
    '''Use this function to preprocess your image (e.g. face crop, alignement, equalization, filtering, etc.)
    '''
    img = bgr_image

    # Align
    img = _faceOperator.find_and_align(img)

    # face not found
    if img is None:
        print("face not found")
        return np.random.randint(255, size=(224, 224), dtype=np.uint8)

    # denoising gaussian
    img = cv2.fastNlMeansDenoisingColored(img, None)

    # gray scale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # equalized
    img = _clahe.apply(img)

    # Sharpening filter
    img = _sharpeningFilter(img)

    return img


def feature_extraction(X:np.ndarray, model=None, outliers=None):
    '''Use this function to extract features from the train and validation sets. 
       Use the model parameter to load a pre-trained feature extractor.
    '''
    # landmarks = fl.get_landmarks(X)
    cld = _faceOperator.calculate_landmarks_distances
    X = np.array([cld(p, index=i, outliers=outliers) for i, p in enumerate(X)])

    if outliers is not None:
        X = np.delete(X, outliers, axis=0)

    if model is None:
        model = PCA()
        model.fit(X)
    X_features = model(X)
    return X_features, model


if __name__ == '__main__':

    # Load the dataset
    path = 'dataset'
    dataset_files = get_dataset_filelist(path)

    # Load the encoder
    labeler = Labeler(dataset_files)

    labeler_path = os.path.join(MODELS_DIR, LABELS_FILENAME)
    if not os.path.exists(labeler_path):
        print("WARNING: A new labels file has been created. Check the working directory.")
        labeler.save(output_path=labeler_path)
    else:
        labeler.load(input_path=labeler_path)

    # Load the files and apply the preprocessing function
    X_train, y_train, X_val, y_val = load_features(
        path, dataset_files, labeler, preprocessing)

    # Compute features
    ''' Loading a feature extraction model if already trained.
        with open('features_model.pkl','rb') as file:
            feature_extration_model = pk.load(file)

         X, _ = feature_extraction(X_train, feature_extration_model) 
    '''
    outliers = []
    print("Extracting features")
    X, pca = feature_extraction(X_train, outliers=outliers)
    print("Extracting features")
    Xv, _ = feature_extraction(X_val, pca)

    # Removing y outliers
    y_train = np.delete(y_train, outliers, axis=0)

    # Save feature extractor
    with open(os.path.join(MODELS_DIR, FEATURES_MODEL), 'wb') as file:
        pk.dump(pca, file)

    # Define a Face Recognition model
    model = FaceRecognition()

    # Train the model
    print("Training the model")
    model.fit(X, y_train, Xv, y_val, tuning_rounds=100)

    # Save the model
    model.save()
