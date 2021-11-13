import os
import numpy as np
import pickle as pk

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from utils import (
    get_dataset_filelist,
    load_files,
    Labeler
)

from train_model import (
    FaceRecognition,
    preprocessing,
    feature_extraction,
    MODELS_DIR,
    LABELS_FILENAME,
    FEATURES_MODEL
)

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
    evaluation_set = 'val'
    evaluation_path = os.path.join(path, evaluation_set)
    X_test, y_test = load_files(
        evaluation_path, dataset_files[evaluation_set], labeler, preprocessing)

    # Compute features
    with open(os.path.join(MODELS_DIR, FEATURES_MODEL), 'rb') as file:
        feature_extration_model = pk.load(file)
    Xt, _ = feature_extraction(X_test, model=feature_extration_model)

    # Define a Face Recognition model
    model = FaceRecognition()

    # Load the model
    model.load()

    # Predict over the evaluation set
    y_pred = model.predict(Xt)

    # Compute the general performance
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy score: ", acc)

    # Compute rejection performance
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    mask = y_test != -1
    y_rej = np.copy(y_test)
    y_rej[mask] = 1
    y_rej[np.logical_not(mask)] = 0

    mask = y_pred != -1
    y_rej_pred = np.copy(y_pred)
    y_rej_pred[mask] = 1
    y_rej_pred[np.logical_not(mask)] = 0

    f1score = f1_score(y_rej, y_rej_pred, zero_division=1)
    precision = precision_score(y_rej, y_rej_pred, zero_division=1)
    recall = recall_score(y_rej, y_rej_pred, zero_division=1)
    print("F1 Score (considering just known/unknown): ", f1score)
    print("Precision (considering just known/unknown): ", precision)
    print("Recall (considering just known/unknown): ", recall)
