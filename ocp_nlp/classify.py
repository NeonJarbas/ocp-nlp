from os import makedirs
from os.path import join, dirname

import joblib
import numpy as np
import random
from sklearn.svm import SVC

from ocp_nlp.features import MediaFeaturesVectorizer
from ovos_classifiers.skovos.tagger import SklearnOVOSClassifier


class MediaTypeClassifier:
    def __init__(self, lang="en"):
        self.lang = lang
        clf = SVC(kernel='linear', probability=True)  # 0.8690685413005272
        self.clf = SklearnOVOSClassifier("cv2", clf)

    def train(self, csv_path, model_folder=None):
        model_folder = model_folder or f"{dirname(__file__)}/models"
        makedirs(model_folder, exist_ok=True)

        with open(csv_path) as f:
            lines = f.read().split("\n")[1:]
            random.shuffle(lines)
            lines = [l.split(",") for l in lines if len(l.split(",")) == 2]
            random.shuffle(lines)

        thresh = int(0.8 * len(lines))
        train = lines[:thresh]
        test = lines[thresh:]
        X = [_[1] for _ in train]
        X_test = [_[1] for _ in test]
        y = [_[0] for _ in train]
        y_test = [_[0] for _ in test]

        self.clf.train(X, y)

        print('Training completed')

        acc = self.clf.score(X_test, y_test)

        print("Accuracy:", acc)
        # Accuracy:  0.7473269555430501

        # save pickle
        path = join(model_folder, f"cv2_svc_media_type_{self.lang}.clf")
        self.clf.save(path)

        # predictions = self.clf.predict(X_test)
        # cm = confusion_matrix(y_test, predictions, labels=self.clf.clf.classes_)
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.clf.clf.classes_)
        # disp.plot()
        # import matplotlib.pyplot as plt
        # plt.savefig('simple_cm.png')
        return acc

    def load(self, model_folder=None):
        model_folder = model_folder or f"{dirname(__file__)}/models"
        path = join(model_folder, f"cv2_svc_media_type_{self.lang}.clf")
        self.clf.load_from_file(path)

    def predict(self, utterances):
        return self.clf.predict(utterances)

    def predict_prob(self, utterances):
        return list(zip(self.clf.predict(utterances),
                        self.clf.predict_proba(utterances)))

    def transform(self, utterances):
        # provide a vector of probabilities per class
        return self.clf.clf.predict_proba(utterances)


class BiasedMediaTypeClassifier:
    def __init__(self, base_clf: MediaTypeClassifier, lang="en", preload=False, dataset_path=None):
        self.lang = lang
        self.base_clf = base_clf
        self.feats = MediaFeaturesVectorizer(lang=self.lang, preload=preload, dataset_path=dataset_path)
        self.clf = SVC(kernel='linear', probability=True)  # 0.9859402460456942

    def register_entity(self, name, samples):
        self.feats.register_entity(name, samples)

    def transform(self, X):
        feats1 = self.feats.transform(X)
        feats2 = self.base_clf.transform(X)
        X2 = []
        for f1, f2 in zip(feats1, feats2):
            f = np.hstack((f1, f2))
            X2.append(f)
        return np.array(X2)

    def train(self, csv_path, model_folder=None):
        model_folder = model_folder or f"{dirname(__file__)}/models"
        makedirs(model_folder, exist_ok=True)

        with open(csv_path) as f:
            lines = f.read().split("\n")[1:]
            random.shuffle(lines)
            lines = [l.split(",") for l in lines if len(l.split(",")) == 2]
            random.shuffle(lines)

        thresh = int(0.8 * len(lines))
        train = lines[:thresh]
        test = lines[thresh:]
        X = [_[1] for _ in train]
        X_test = [_[1] for _ in test]
        y = [_[0] for _ in train]
        y_test = [_[0] for _ in test]

        X = self.transform(X)
        self.clf.fit(X, y)

        print('Training completed')

        # save pickle
        path = join(model_folder, f"biased_svc_media_type_{self.lang}.clf")
        joblib.dump(self.clf, path)
        # self.clf.save(path)

        X_test = self.transform(X_test)
        acc = self.clf.score(X_test, y_test)

        print("Accuracy:", acc)
        # Accuracy:  0.88

        # import matplotlib.pyplot as plt
        # predictions = self.clf.predict(X_test)
        # cm = confusion_matrix(y_test, predictions, labels=self.clf.classes_)
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.clf.classes_)
        # disp.plot()
        # plt.show()

        return acc

    def precompute_features(self, csv_path, model_folder=None):
        model_folder = model_folder or f"{dirname(__file__)}/models"
        makedirs(model_folder, exist_ok=True)

        with open(csv_path) as f:
            lines = f.read().split("\n")[1:]
            random.shuffle(lines)
            lines = [l.split(",") for l in lines if len(l.split(",")) == 2]
            random.shuffle(lines)

        X = [_[1] for _ in lines]
        y = [_[0] for _ in lines]

        print('computing features, this might take a while')

        X = self.transform(X)

        print('Features computed')

        # save pickle
        xpath = join(model_folder, f"media_type_{self.lang}.X")
        joblib.dump(X, xpath)
        ypath = join(model_folder, f"media_type_{self.lang}.y")
        joblib.dump(y, ypath)
        return xpath, ypath

    def train_from_precomputed(self, model_folder=None):
        model_folder = model_folder or f"{dirname(__file__)}/models"

        print("loading pre computed features")

        xpath = join(model_folder, f"media_type_{self.lang}.X")
        X = joblib.load(xpath)
        ypath = join(model_folder, f"media_type_{self.lang}.y")
        y = joblib.load(ypath)

        thresh = int(0.8 * len(X))
        X_test = X[thresh:]
        y_test = y[thresh:]
        X = X[:thresh]
        y = y[:thresh]

        self.clf.fit(X, y)

        print('Training completed')

        # save pickle
        path = join(model_folder, f"biased_svc_media_type_{self.lang}.clf")
        joblib.dump(self.clf, path)
        # self.clf.save(path)

        acc = self.clf.score(X_test, y_test)

        print("Accuracy:", acc)
        # Accuracy:  0.88

        # predictions = self.clf.predict(X_test)
        # cm = confusion_matrix(y_test, predictions, labels=self.clf.classes_)
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.clf.classes_)
        # disp.plot()
        # import matplotlib.pyplot as plt
        # plt.savefig('biased_cm.png')

        return acc

    def load(self, model_file=None):
        path = model_file or f"{dirname(__file__)}/models/biased_svc_media_type_{self.lang}.clf"
        self.clf = joblib.load(path)

    def predict(self, utterances):
        X = self.transform(utterances)
        return self.clf.predict(X)

    def predict_prob(self, utterances):
        X = self.transform(utterances)
        p = np.max(self.clf.predict_proba(X), axis=1)
        return list(zip(self.clf.predict(X), p))


if __name__ == "__main__":
    # download dataset from https://github.com/NeonJarbas/OCP-dataset
    csv_path = f"{dirname(__file__)}/dataset.csv"
    entities_path = f"{dirname(__file__)}/sparql_ocp"

    # basic text only classifier
    clf1 = MediaTypeClassifier()
    clf1.train(csv_path)  # Accuracy: 0.7473269555430501
    clf1.load()

    label, confidence = clf1.predict_prob(["play metallica"])[0]
    print(label, confidence)  # [('music', 0.3438956411030462)]

    # keyword biased classifier, uses the above internally for extra features
    clf = BiasedMediaTypeClassifier(clf1, lang="en", preload=True, dataset_path=entities_path)  # load entities database

    # csv_path = f"{dirname(__file__)}/sparql_ocp/dataset.csv"
    # clf.precompute_features(csv_path)
    clf.train_from_precomputed()  # Accuracy: 0.8868880135059088

    clf.load()

    # klownevilus is an unknown entity
    label, confidence = clf.predict_prob(["play klownevilus"])[0]
    print(label, confidence)  # music 0.3398020446925623

    # probability increases for movie
    clf.register_entity("movie_name", ["klownevilus"])  # movie correctly predicted now
    label, confidence = clf.predict_prob(["play klownevilus"])[0]
    print(label, confidence)  # movie 0.540225616798516
