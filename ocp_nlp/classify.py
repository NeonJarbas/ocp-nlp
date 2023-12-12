import random
from os import makedirs
from os.path import join, dirname

import joblib
import numpy as np
from ovos_classifiers.skovos.tagger import SklearnOVOSClassifier
from sklearn.svm import SVC

from ocp_nlp.features import MediaFeaturesVectorizer
from ocp_nlp.constants import MediaType
from ocp_nlp.utils import ParallelWorkers


class _OCPClassifier:
    def __init__(self, model_name, lang="en"):
        self.model_name = model_name
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
        path = join(model_folder, f"{self.model_name}_{self.lang}.clf")
        self.clf.save(path)

        # from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        # predictions = self.clf.predict(X_test)
        # cm = confusion_matrix(y_test, predictions, labels=self.clf.clf.classes_)
        # disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.clf.clf.classes_)
        # disp.plot()
        # import matplotlib.pyplot as plt
        # plt.savefig(f'{self.model_name}_cm.png')
        return acc

    def load(self, model_folder=None):
        model_folder = model_folder or f"{dirname(__file__)}/models"
        path = join(model_folder, f"{self.model_name}_{self.lang}.clf")
        self.clf.load_from_file(path)

    def predict(self, utterances):
        return self.clf.predict(utterances)

    def predict_prob(self, utterances):
        return list(zip(self.clf.predict(utterances),
                        self.clf.predict_proba(utterances)))

    def transform(self, utterances):
        # provide a vector of probabilities per class
        return self.clf.clf.predict_proba(utterances)


class MediaTypeClassifier(_OCPClassifier):
    def __init__(self, lang="en"):
        super().__init__("cv2_svc_media_type", lang)


class BiasedMediaTypeClassifier:
    def __init__(self, base_clf: MediaTypeClassifier, lang="en", preload=False, dataset_path=None):
        self.lang = lang
        self.base_clf = base_clf  # 0.7597073719752392
        self.feats = MediaFeaturesVectorizer(lang=self.lang, preload=preload, dataset_path=dataset_path)
        self.clf = SVC(kernel='linear', probability=True)  # 0.8868880135059088

    def extract_entities(self, utterance):
        return self.feats._transformer.wordlist.extract(utterance)

    @staticmethod
    def label2media(label):
        if label == "ad":
            mt = MediaType.AUDIO_DESCRIPTION
        elif label == "adult":
            mt = MediaType.ADULT
        elif label == "adult_asmr":
            mt = MediaType.ADULT_AUDIO
        elif label == "anime":
            mt = MediaType.ANIME
        elif label == "audio":
            mt = MediaType.AUDIO
        elif label == "audiobook":
            mt = MediaType.AUDIOBOOK
        elif label == "bts":
            mt = MediaType.BEHIND_THE_SCENES
        elif label == "bw_movie":
            mt = MediaType.BLACK_WHITE_MOVIE
        elif label == "cartoon":
            mt = MediaType.CARTOON
        elif label == "comic":
            mt = MediaType.VISUAL_STORY
        elif label == "documentary":
            mt = MediaType.DOCUMENTARY
        elif label == "game":
            mt = MediaType.GAME
        elif label == "hentai":
            mt = MediaType.HENTAI
        elif label == "movie":
            mt = MediaType.MOVIE
        elif label == "music":
            mt = MediaType.MUSIC
        elif label == "news":
            mt = MediaType.NEWS
        elif label == "podcast":
            mt = MediaType.PODCAST
        elif label == "radio":
            mt = MediaType.RADIO
        elif label == "radio_drama":
            mt = MediaType.RADIO_THEATRE
        elif label == "series":
            mt = MediaType.VIDEO_EPISODES
        elif label == "short_film":
            mt = MediaType.SHORT_FILM
        elif label == "silent_movie":
            mt = MediaType.SILENT_MOVIE
        elif label == "trailer":
            mt = MediaType.TRAILER
        elif label == "tv_channel":
            mt = MediaType.TV
        elif label == "video":
            mt = MediaType.VIDEO
        else:
            mt = MediaType.GENERIC
        return mt

    def register_entity(self, name, samples):
        self.feats.register_entity(name, samples)

    def transform(self, X):
        if isinstance(X, str):
            X = [X]
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

    def transform_from_precomputed(self, X, tagged_sents):
        feats1 = []
        for s in X:
            if s not in tagged_sents:
                f = self.feats.transform([s])[0]
            else:
                f = tagged_sents[s]
            feats1.append(f)

        feats2 = self.base_clf.transform(X)
        X2 = []
        for f1, f2 in zip(feats1, feats2):
            f = np.hstack((f1, f2))
            X2.append(f)
        return np.array(X2)

    def precompute_features(self, csv_path, model_folder=None):
        model_folder = model_folder or f"{dirname(__file__)}/models"
        makedirs(model_folder, exist_ok=True)

        with open(csv_path) as f:
            lines = f.read().split("\n")[1:]
            random.shuffle(lines)
            lines = [l.split(",") for l in lines if len(l.split(",")) == 2]
            random.shuffle(lines)

        X = [_[1] for _ in lines]

        print('tagging dataset, this might take a while')

        def heavy_work(u):
            f = self.feats.transform([u])[0]
            return f

        t = ParallelWorkers()
        # t.workers = 26
        tagged_sents = t.do_work(X, heavy_work)

        print('dataset tagged')

        # save pickle
        xpath = join(model_folder, f"tagged_sentences.X")
        joblib.dump(tagged_sents, xpath)
        return xpath

    def train_from_precomputed(self, csv_path, model_folder=None):
        model_folder = model_folder or f"{dirname(__file__)}/models"

        print("loading pre computed features")

        path = join(model_folder, f"tagged_sentences.X")
        tagged_sents = joblib.load(path)

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

        X = self.transform_from_precomputed(X, tagged_sents)

        self.clf.fit(X, y)

        print('Training completed')

        # save pickle
        path = join(model_folder, f"biased_svc_media_type_{self.lang}.clf")
        joblib.dump(self.clf, path)

        X_test = self.transform_from_precomputed(X_test, tagged_sents)
        acc = self.clf.score(X_test, y_test)

        print("Accuracy:", acc)
        # Accuracy:  0.88

        #from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
        #predictions = self.clf.predict(X_test)
        #cm = confusion_matrix(y_test, predictions, labels=self.clf.classes_)
        #disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.clf.classes_)
        #disp.plot()
        #import matplotlib.pyplot as plt
        #plt.savefig('biased_cm.png')

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


class BinaryPlaybackClassifier(_OCPClassifier):
    def __init__(self, lang="en"):
        super().__init__("cv2_svc_binary_ocp", lang)


if __name__ == "__main__":
    # download datasets from https://github.com/NeonJarbas/OCP-dataset

    csv_path = f"/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/playback.csv"

    o = BinaryPlaybackClassifier()
    # o.train(csv_path)  # 0.9915889974994316
    o.load()

    preds = o.predict(["play a song", "play my morning jams",
                       "i want to watch the matrix",
                       "tell me a joke", "who are you", "you suck"])
    print(preds)  # ['OCP' 'OCP' 'OCP' 'other' 'other' 'other']

    csv_path = f"/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/dataset.csv"
    entities_path = f"/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/sparql_ocp"

    # basic text only classifier
    clf1 = MediaTypeClassifier()
    # clf1.train(csv_path)  # Accuracy: 0.8489847715736041
    clf1.load()

    label, confidence = clf1.predict_prob(["play metallica"])[0]
    print(label, confidence)  # [('music', 0.3818757631643521)]

    # keyword biased classifier, uses the above internally for extra features
    clf = BiasedMediaTypeClassifier(clf1, lang="en",
                                    preload=True, dataset_path=entities_path)  # load entities database

    #clf.precompute_features(csv_path)

    clf.train_from_precomputed(csv_path)  # Accuracy: 0.9809644670050761
    #clf.load()

    # klownevilus is an unknown entity
    label, confidence = clf.predict_prob(["play klownevilus"])[0]
    print(label, confidence)  # music 0.2517992708279099

    # probability increases for movie
    clf.register_entity("movie_name", ["klownevilus"])  # movie correctly predicted now
    label, confidence = clf.predict_prob(["play klownevilus"])[0]
    print(label, confidence)  # movie 0.3957754156588387
