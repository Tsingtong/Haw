# -*- coding: utf-8 -*-

import os
import os.path
import shutil

import face_recognition
import pickle
import numpy as np
import pandas as pd
from src.haw_knn_utils import knn_model
from face_recognition.face_recognition_cli import image_files_in_folder


def _organize_images_into_folders(datadir):
    """
    This is a local function, to move images into folder which will be create with it's own name

    :param datadir:
    :return: None
    """
    # Loop through each image to create folder
    for each_image in os.listdir(datadir):
        if each_image == '.DS_Store':
            continue
        # Create folder
        image_abs_name = each_image.rstrip(".JPG")
        image_abs_name = image_abs_name.rstrip(".jpg")
        image_path = os.path.join(datadir, each_image)
        fold_path = os.path.join(datadir, image_abs_name)
        os.makedirs(fold_path)
        # Move file into corresponding folder
        shutil.move(src=image_path, dst=fold_path)


def register_face(datadir, outputdir):
    """
    verbose: verbosity of person

    :param datadir:
    :param outputdir:
    :return: True
    """
    verbose = False
    face_set = np.empty((128, 0), dtype='float64')
    name_set = []
    # Loop through each person in the data set(face) [Read to Memory]
    for class_dir in os.listdir(datadir):
        if not os.path.isdir(os.path.join(datadir, class_dir)):
            continue
        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(datadir, class_dir)):
            image = face_recognition.load_image_file(img_path)
            face_bounding_boxes = face_recognition.face_locations(image)

            if len(face_bounding_boxes) != 1:
                # If there are no people (or too many people) in a training image, stop register process.
                if verbose:
                    print("Image {} not suitable for encoding.Reason: {}".format(img_path, "Didn't find a face" if len(
                        face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                # Add face encoding for current image to the ndarray set
                face_coding = face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0]
                face_set = np.column_stack((face_set, face_coding))
                name_set.append(class_dir)

    # Save dat into bin file(ndarray) and txt(list)
    np.save(os.path.join(outputdir, "face_set.npy"), face_set)
    print('saved face_set:', np.shape(face_set))
    df = pd.DataFrame(name_set)
    df.to_csv(os.path.join(outputdir, "name_set.csv"), index=False, sep=',', header=None)
    return True


def load_face_coding(datadir):
    """
    In this function, program will load face coding into memory which is stored in the Global object : g

    :param datadir:
    :return: Face set(array)[(128,n)] / Name set(list)[name1,name2,name....]
    """
    face_set = np.empty((128, 0), dtype='float64')
    name_set = []
    # Load ndarray from bin file
    face_set = np.load(os.path.join(datadir, "face_set.npy"))
    print('loaded face_set:', np.shape(face_set))
    # Load name from csv file
    df = pd.read_csv(os.path.join(datadir, "name_set.csv"), header=None)
    name_array = np.array(df)  # np.ndarray()
    name_set = name_array.tolist()  # list
    return face_set, name_set


def train_model(train_data_dir, model_save_path, n_neighbors):
    """
    In this function, knn classifier will be trained, which is based on Face Set(array type)

    :param train_data_dir:
    :param model_save_path:
    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
    :return: A trained classifier
    """
    print("Training KNN classifier...")
    classifier = knn_model.train(train_dir=train_data_dir,
                                       model_save_path=os.path.join(model_save_path, "trained_knn_model.clf"),
                                       n_neighbors=n_neighbors)
    print("Training complete!")
    return classifier


def load_trained_model(model_save_path):
    """
    Load trained knn model to memory

    :param model_save_path:
    :return: A trained classifier
    """
    if model_save_path is None:
        raise Exception("Must supply knn classifier path : model_path")
    model_save_path = os.path.join(model_save_path, 'trained_knn_model.clf')
    # Load a trained KNN model (if one was passed in)
    knn_clf = None
    with open(model_save_path, 'rb') as f:
        knn_clf = pickle.load(f)
    return knn_clf


def facerec_from_image(img_path, knn_clf, distance_threshold=0.6, show_image_in_window=False):
    """
    Use face_recognition's API and combine KNN classify function to achieve face recognition goal.

    :param img_path:
    :param knn_clf: trained knn model
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :param show_image_in_window: (optional)(In developing) default is False. Unfinished function.
    :return: faces_encodings, face_locations, are_matches(tuple)
    """
    if show_image_in_window:
        pass
    else:
        # Load image file and find face locations
        image = face_recognition.load_image_file(img_path)
        face_locations = face_recognition.face_locations(image)

        # If no faces are found in the image, return an empty result.
        if len(face_locations) == 0:
            return []

        # Find encodings for faces in the test iamge
        faces_encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)

        # Use the KNN model to find the best matches for the test face
        closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
        are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(face_locations))]

        # Predict classes and remove classifications that aren't within the threshold
        return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
                zip(knn_clf.predict(faces_encodings), face_locations, are_matches)]


def facerec_for_webservice(img, knn_clf, distance_threshold=0.6):
    """
    Overwrite knn_model's predict function to provide face recognition for haw_inquire_service.
    Why I write this: To improve program scalability, I separate `facerec_from_image`&`facerec_for_webservice`
                      into two functions. So that, We can rewrite function without bad influence.

    :param img:
    :param knn_clf: trained knn model
    :param distance_threshold:(optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: faces_encodings, face_locations, are_matches(tuple)
    """
    # Load image file and find face locations
    image = face_recognition.load_image_file(img)
    face_locations = face_recognition.face_locations(image)

    # If no faces are found in the image, return an empty result.
    if len(face_locations) == 0:
        return []

    # Find encodings for faces in the test iamge
    faces_encodings = face_recognition.face_encodings(image, known_face_locations=face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
            zip(knn_clf.predict(faces_encodings), face_locations, are_matches)]


if __name__ == '__main__':
    # test module
    datadir = '/Users/liuqingtong/Desktop/PycharmProjects/Face_Rec_BTBU_v1.0/data/17_/'
    outputdir = '/Users/liuqingtong/Desktop/PycharmProjects/Face_Rec_BTBU_v1.0/FaceDB/'
    # _organize_images_into_folders(datadir)
    # register_face(datadir, outputdir)
    # load_face_coding(outputdir)
