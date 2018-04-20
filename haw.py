# -*- coding: utf-8 -*-

import argparse
from src.haw_utils import *
from src.haw_web_service_utils import haw_inquire_service
from src.haw_videorec_utils import videorec_faster


class Global:
    "Global-Class's WorkType: 'register_face'、'load_face'"

    face_db_coding = []
    face_db_name = []
    classifier = None

    def __init__(self, HawType, DataDir, OutputDir):
        self.DataDir = DataDir
        self.OutputDir = OutputDir
        self.HawType = HawType

    def getDataDir(self):
        return self.DataDir

    def getOutputDir(self):
        return self.OutputDir

    def getHawType(self):
        return self.HawType

    def getClassifier(self):
        return self.classifier

    def loadFaceSet_to_Memory(self, face_set, name_set):
        self.face_db_coding = face_set
        self.face_db_name = name_set
        return True

    def storeClassifier(self, classifier):
        self.classifier = classifier
        return True

    def loadClassifier(self, saved_model_path):
        self.classifier = load_trained_model(g.getOutputDir())
        return classifier


def _init():
    # parse args and create Global object
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", help="define your DataSet directory location", type=str)
    parser.add_argument("--output_dir", help="define your npy output directory location", type=str)
    parser.add_argument("--haw_type", help="define haw work type,register_face,load_face", type=str)

    args = parser.parse_args()
    global g
    g = Global(HawType=args.haw_type, DataDir=args.data_dir, OutputDir=args.output_dir)


def loadClassifier():
    classifier = load_trained_model(g.getOutputDir())
    return classifier


if __name__ == '__main__':
    _init()
    if g.getHawType().find('register_face') != -1:
        if register_face(g.getDataDir(), g.getOutputDir()):
            print('Registered')

    if g.getHawType().find('load_face') != -1:
        face_set, name_set = load_face_coding(g.getDataDir())
        g.loadFaceSet_to_Memory(face_set, name_set)
        print('Load to memory progress done')

    if g.getHawType().find('train_knn_model') != -1:
        classifier = train_model(train_data_dir=g.getDataDir(), model_save_path=g.getOutputDir(), n_neighbors=2)
        if g.storeClassifier(classifier):
            print('Train model progress done!')

    if g.getHawType().find('predict_image') != -1:
        knn_clf = load_trained_model(model_save_path=g.getOutputDir())
        predictions = facerec_from_image(img_path=g.getDataDir(), knn_clf=knn_clf)
        # Print results on the console
        for name, (top, right, bottom, left) in predictions:
            print("- Found {} at ({}, {})".format(name, left, top))
        print('Predict progress done!')

    if g.getHawType().find('up_webservice') != -1:
        haw_inquire_service.run()

    if g.getHawType().find('webcam_recognition') != -1:
        videorec_faster.run()