import os.path
from os import makedirs
from os.path import join, dirname

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, Perceptron, PassiveAggressiveClassifier
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC

from ocp_nlp.constants import MediaType
from ocp_nlp.features import MediaFeaturesVectorizer, BiasFeaturesVectorizer
from ovos_classifiers.skovos.tagger import SklearnOVOSClassifier


def iter_clfs():
    clfs = {
        "lsvc": LinearSVC(),
        "percep": Perceptron(),
        "pa": PassiveAggressiveClassifier(),
        "et": ExtraTreesClassifier(n_estimators=50),
        "mlp": MLPClassifier(max_iter=1000),
        "svc": SVC(kernel='linear', probability=True),
        "lr": LogisticRegression(),
        "mnb": MultinomialNB(),
        #  "gnb": GaussianNB(),
        #  "bnb": BernoulliNB(),
        "rf": RandomForestClassifier()
    }
    featselecters = {
        "fs_lsvc_p": Pipeline([
            ('feature_selection', SelectFromModel(LinearSVC(dual="auto", penalty="l1"))),
            ('classification', Perceptron())
        ]),
        "fs_lsvc_lr": Pipeline([
            ('feature_selection', SelectFromModel(LinearSVC(dual="auto", penalty="l1"))),
            ('classification', LogisticRegression())
        ]),
        "fs_lsvc_svc": Pipeline([
            ('feature_selection', SelectFromModel(LinearSVC(dual="auto", penalty="l1"))),
            ('classification', SVC(kernel='linear', probability=True))
        ]),
        "fs_lsvc_mlp": Pipeline([
            ('feature_selection', SelectFromModel(LinearSVC(dual="auto", penalty="l1"))),
            ('classification', MLPClassifier(max_iter=300))
        ])
    }
    votings = {
        "v_svcmnbmlp": VotingClassifier(estimators=[("SVC", SVC(kernel='linear', probability=True)),
                                                    ("MultinomialNB", MultinomialNB()),
                                                    ("MLPClassifier", MLPClassifier(max_iter=300))
                                                    ], voting="soft"),
        "v_svclrmlp": VotingClassifier(estimators=[("SVC", SVC(kernel='linear', probability=True)),
                                                   ("LogisticRegression", LogisticRegression()),
                                                   ("MLPClassifier", MLPClassifier(max_iter=300))
                                                   ], voting="soft"),
        "v_svccplr": VotingClassifier(estimators=[("SVC", SVC(kernel='linear', probability=True)),
                                                  ("CalibratedPerceptron", CalibratedClassifierCV(Perceptron())),
                                                  ("LogisticRegression", LogisticRegression())
                                                  ], voting="soft"),
        "v_svccplrmnb": VotingClassifier(estimators=[("SVC", SVC(kernel='linear', probability=True)),
                                                     ("CalibratedPerceptron", CalibratedClassifierCV(Perceptron())),
                                                     ("LogisticRegression", LogisticRegression()),
                                                     ("MultinomialNB", MultinomialNB()),
                                                     ], voting="soft")
    }
    calibrated = {f"c_{k}": CalibratedClassifierCV(clf)
                  for k, clf in clfs.items()}

    for k, clf in calibrated.items():
        yield k, clf


class _OCPClassifier:
    def __init__(self, model_name, lang="en"):
        self.model_name = model_name
        self.lang = lang
        self.clf = SklearnOVOSClassifier("cv2", None)

    def split_train_test(self, csv_path, test_size=0.6):

        with open(csv_path) as f:
            lines = f.read().split("\n")[1:]
            lines = [l.split(",", 1) for l in lines if "," in l]

        X = [_[1] for _ in lines]
        y = [_[0] for _ in lines]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                            random_state=42, stratify=y)
        return X_train, X_test, y_train, y_test

    def search_best(self, csv_path, model_folder=None, retrain=False):
        model_folder = model_folder or f"{dirname(__file__)}/models"
        makedirs(model_folder, exist_ok=True)

        X_train, X_test, y_train, y_test = self.split_train_test(csv_path, test_size=0.6)

        best = (None, 0)
        for k, c in iter_clfs():
            path = join(model_folder, f"{self.model_name}_{self.lang}.{k}")
            if os.path.isfile(path) and not retrain:
                continue

            clf = SklearnOVOSClassifier("cv2", c)
            try:
                clf.train(X_train, y_train)
            except:
                continue
            print(c, 'Training completed')

            y_pred = clf.predict(X_test)
            acc = balanced_accuracy_score(y_test, y_pred)

            print(c, "Accuracy:", acc)

            report = classification_report(y_test, y_pred, target_names=c.classes_)
            print(report)
            with open(f'{model_folder}/reports/{self.model_name}_{k}.txt', "w") as f:
                f.write(report)

            if acc > best[1]:
                best = (clf, acc)
            # save pickle
            clf.save(path)

            try:
                cm = confusion_matrix(y_test, y_pred, labels=c.classes_)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=c.classes_)
                disp.plot()
                import matplotlib.pyplot as plt
                plt.savefig(f'{model_folder}/reports/{self.model_name}_{k}_cm.png')
            except:
                pass

        print("BEST", best[0].clf, "Accuracy", best[1])
        self.clf = best[0]
        return best

    def load(self, model_path=None):
        path = model_path or f"{dirname(__file__)}/models/{self.model_name}_{self.lang}.c_percep"
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
        super().__init__("cv2_media_type", lang)

    def load(self, model_path=None):
        model_path = model_path or f"{dirname(__file__)}/models/{self.model_name}_{self.lang}.c_percep"
        super().load(model_path)


class BinaryPlaybackClassifier(_OCPClassifier):
    def __init__(self, lang="en"):
        super().__init__("cv2_binary_ocp", lang)


class KeywordMediaTypeClassifier:
    def __init__(self, lang="en", preload=True, model_name="kword_media_type", entities_path=None):
        self.lang = lang
        self.feats = MediaFeaturesVectorizer(preload=preload, dataset_path=entities_path)
        self.feats2 = BiasFeaturesVectorizer(preload=preload, dataset_path=entities_path)
        self.clf = CalibratedClassifierCV(SVC(kernel='linear', probability=True))  # 0.9806825499034127
        self.model_name = model_name

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
        self.feats2.register_entity(name, samples)

    def transform(self, X):
        if isinstance(X, str):
            X = [X]
        feats1 = self.feats.transform(X)
        feats2 = self.feats2.transform(X)
        X2 = []
        for f1, f2 in zip(feats1, feats2):
            f = np.hstack((f1, f2))
            X2.append(f)
        return np.array(X2)

    def split_train_test(self, csv_path, test_size=0.6):

        with open(csv_path) as f:
            lines = f.read().split("\n")[1:]
            lines = [l.split(",", 1) for l in lines if "," in l]

        X = [_[1] for _ in lines]
        y = [_[0] for _ in lines]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                            random_state=42, stratify=y)
        return X_train, X_test, y_train, y_test

    def search_best(self, csv_path, model_folder=None, retrain=False):
        model_folder = model_folder or f"{dirname(__file__)}/models"
        makedirs(model_folder, exist_ok=True)

        X_train, X_test, y_train, y_test = self.split_train_test(csv_path, test_size=0.6)
        X_train = self.transform(X_train)
        X_test = self.transform(X_test)

        best = (None, 0)
        for k, clf in iter_clfs():
            path = join(model_folder, f"{self.model_name}_{self.lang}.{k}")
            if os.path.isfile(path) and not retrain:
                continue

            clf.fit(X_train, y_train)

            print(clf, 'Training completed')

            y_pred = clf.predict(X_test)
            acc = balanced_accuracy_score(y_test, y_pred)

            print(clf, "Accuracy:", acc)

            report = classification_report(y_test, y_pred, target_names=clf.classes_)
            print(report)
            with open(f'{model_folder}/reports/{self.model_name}_{self.lang}_{k}.txt', "w") as f:
                f.write(report)

            if acc > best[1]:
                best = (clf, acc)
            # save pickle
            joblib.dump(clf, path)
            # clf.save(path)

            try:
                cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
                disp.plot()
                import matplotlib.pyplot as plt
                plt.savefig(f'{model_folder}/reports/{self.model_name}_{self.lang}_{k}_cm.png')
            except:
                pass

        print("BEST", best[0], "Accuracy", best[1])
        self.clf = best[0]
        return best

    def load(self, model_file=None):
        path = model_file or f"{dirname(__file__)}/models/{self.model_name}_{self.lang}.c_lsvc"
        self.clf = joblib.load(path)

    def predict(self, utterances):
        X = self.transform(utterances)
        return self.clf.predict(X)

    def predict_prob(self, utterances):
        X = self.transform(utterances)
        p = np.max(self.clf.predict_proba(X), axis=1)
        return list(zip(self.clf.predict(X), p))

    def predict_labels(self, utterances):
        X = self.transform(utterances)
        probs = []
        for pred in self.clf.predict_proba(X):
            classes = self.clf.classes_
            labels = {classes[idx]: p for idx, p in enumerate(pred)}
            probs.append(labels)
        return probs


class BiasedMediaTypeClassifier(KeywordMediaTypeClassifier):
    def __init__(self, base_clf: MediaTypeClassifier = None, lang="en", preload=True,
                 model_name="cv2_biased_media_type", entities_path=None):
        super().__init__(lang, preload, model_name, entities_path)
        if base_clf is None:
            base_clf = MediaTypeClassifier()
            base_clf.load()
        self.base_clf = base_clf

    def load(self, model_path=None):
        model_path = model_path or f"{dirname(__file__)}/models/{self.model_name}_{self.lang}.c_mlp"
        super().load(model_path)

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


if __name__ == "__main__":
    from ovos_utils.log import LOG

    ents_csv_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/ocp_entities_v0.csv"

    LOG.set_level("DEBUG")
    # download datasets from https://github.com/NeonJarbas/OCP-dataset

    csv_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/ocp_sentences_v0.csv"

    o = BinaryPlaybackClassifier()
    #o.search_best(csv_path)  # 0.99
    o.load()

    preds = o.predict(["play a song", "play my morning jams",
                       "i want to watch the matrix",
                       "tell me a joke", "who are you", "you suck"])
    # print(preds)  # ['OCP' 'OCP' 'OCP' 'other' 'other' 'other']

    csv_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/ocp_media_types_v0.csv"

    # basic text only classifier
    clf1 = MediaTypeClassifier()
    # clf1.split_train_test(csv_path)
    #clf1.search_best(csv_path)  # Accuracy: 0.8139614643381273
    # m = "/home/miro/PycharmProjects/OCP_sprint/ocp-nlp/ocp_nlp/models/cv2_media_type_en.c_fs_lsvc_mlp"
    clf1.load()

    label, confidence = clf1.predict_prob(["play metallica"])[0]
    print(label, confidence)  # [('music', 0.15532930055019162)]

    # keyword biased classifier, uses the above internally for extra features
    # clf = KeywordMediaTypeClassifier()
    # clf.search_best(csv_path)  #  Accuracy 0.735218828459692

    clf = BiasedMediaTypeClassifier(lang="en", preload=True,
                                    entities_path=ents_csv_path)  # load entities database
    # clf = KeywordMediaTypeClassifier()
    #clf.search_best(csv_path)  # Accuracy 0.9311456835977218
    # m = "/home/miro/PycharmProjects/OCP_sprint/ocp-nlp/ocp_nlp/models/cv2_biased_media_type_en.c_percep"
    clf.load()

    # klownevilus is an unknown entity
    labels = clf.predict_labels(["play klownevilus"])[0]
    old = labels["movie"]
    old2 = labels["music"]

    # probability increases for movie
    clf.register_entity("movie_name", ["klownevilus"])

    labels2 = clf.predict_labels(["play klownevilus"])[0]
    n = labels2["movie"]
    n2 = labels2["music"]

    print("bias changed movie confidence: ", old, "to", n)
    print("bias changed music confidence: ", old2, "to", n2)
    # bias changed movie confidence:  0.0495770016040964 to 0.42184841516803073
    # bias changed music confidence:  0.20116106628854769 to 0.0876066334326611
    assert n > old
    assert n2 < old2

    from pprint import pprint

    pprint(labels)
    pprint(labels2)
