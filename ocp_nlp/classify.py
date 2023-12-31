import os.path
import random
from os import makedirs
from os.path import join, dirname

import numpy as np
from ovos_utils.log import LOG
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import Perceptron
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from ocp_nlp.constants import MediaType
from ocp_nlp.features import MediaFeaturesVectorizer, BiasFeaturesVectorizer, KeywordFeatures, PositiveBias
from ovos_classifiers.skovos.classifier import SklearnOVOSClassifier, iter_clfs


class _OCPClassifier:
    def __init__(self, model_name, lang="en",
                 pipeline_id="cv2"):
        self.model_name = model_name
        self.lang = lang.split("-")[0].lower()
        self.pipeline_id = pipeline_id
        self.clf = SklearnOVOSClassifier(pipeline_id, None)

    def transform(self, X):
        return X

    def split_train_test(self, csv_path, test_size=0.6):

        X, y = self.read_csv(csv_path)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                            random_state=42, stratify=y)
        return X_train, X_test, y_train, y_test

    def read_csv(self, csv_path):
        with open(csv_path) as f:
            lines = f.read().split("\n")[1:]
            lines = [l.split(",", 1) for l in lines if "," in l]

        X = [_[1].strip() for _ in lines]
        y = [_[0].strip() for _ in lines]
        return X, y

    def search_best(self, csv_path, test_csv_path=None, model_folder=None, retrain=False):
        model_folder = model_folder or f"{dirname(__file__)}/models"
        makedirs(model_folder, exist_ok=True)

        if test_csv_path:
            X_train, y_train = self.read_csv(csv_path)
            X_test, y_test = self.read_csv(test_csv_path)
        else:
            X_train, X_test, y_train, y_test = self.split_train_test(csv_path, test_size=0.6)

        X_train = self.transform(X_train)
        X_test = self.transform(X_test)

        best = (None, 0)
        for k, c in list(iter_clfs(calibrate=True, voting=True)) + list(iter_clfs(calibrate=True)):
            path = join(model_folder, f"{self.model_name}_{self.lang}.{k}")
            if os.path.isfile(path) and not retrain:
                continue

            clf = SklearnOVOSClassifier(self.pipeline_id, c)
            try:
                clf.train(X_train, y_train)
            except:
                continue
           # print(c, 'Training completed')

            y_pred = clf.predict(X_test)
            acc = balanced_accuracy_score(y_test, y_pred)

           # print(c)
            report = f"Accuracy: {acc}\n" + classification_report(y_test, y_pred, target_names=c.classes_)
           # print(report)
            with open(f'{model_folder}/reports/{self.model_name}_{self.lang}_{k}.txt', "w") as f:
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

       # print("BEST", best[0].clf, "Accuracy", best[1])
        self.clf = best[0]
        return best

    def load(self, model_path=None):
        path = model_path or f"{dirname(__file__)}/models/{self.model_name}_{self.lang}.c_percep"
        self.clf.load_from_file(path)

    def predict(self, utterances, probability=False):
        X = self.transform(utterances)
        if probability:
            return list(zip(self.clf.predict(X),
                            self.clf.predict_proba(X)))
        return self.clf.predict(X)

    def predict_labels(self, utterances):
        X = self.transform(utterances)
        return self.clf.predict_labels(X)

    def vectorize(self, utterances):
        X = self.transform(utterances)
        # provide a vector of probabilities per class
        return self.clf.clf.predict_proba(X)


# word features only - lang specific
class PlaybackTypeClassifier(_OCPClassifier):
    """
        Balanced Accuracy: 0.9051382586670508

                  precision    recall  f1-score   support

           audio       0.94      0.94      0.94      2581
        external       0.98      0.82      0.89       240
           video       0.95      0.95      0.95      3328

        accuracy                           0.94      6149
       macro avg       0.95      0.91      0.93      6149
    weighted avg       0.95      0.94      0.94      6149
    """

    def __init__(self, model_name="cv2_playback_type", lang="en"):
        super().__init__(model_name, lang)

    def load(self, model_path=None):
        model_path = model_path or f"{dirname(__file__)}/models/{self.model_name}_{self.lang}.c_percep"
        super().load(model_path)


class MediaTypeClassifier(_OCPClassifier):
    def __init__(self, model_name="cv2_media_type", lang="en"):
        super().__init__(model_name, lang)

    def load(self, model_path=None):
        model_path = model_path or f"{dirname(__file__)}/models/{self.model_name}_{self.lang}.c_percep"
        super().load(model_path)


class BinaryPlaybackClassifier(_OCPClassifier):
    def __init__(self, lang="en"):
        super().__init__("cv2_binary_ocp", lang)


# using keyword features - lang agnostic
class BaseKeywordClassifier(_OCPClassifier):
    def __init__(self, lang="all", preload=True, model_name="kword_biased_media_type",
                 enabled_features=None):
        self.feats = MediaFeaturesVectorizer(preload=preload)
        self.feats2 = BiasFeaturesVectorizer(preload=preload)
        super().__init__(model_name, lang, pipeline_id="raw")
        # store in self.clf so it gets saved to pickle
        self.clf.enabled_feats = enabled_features or ["keyword", "bias"]

    def find_best_Perceptron(self, csv_path, test_csv_path=None, model_folder=None):
        model_folder = model_folder or f"{dirname(__file__)}/models"
        makedirs(model_folder, exist_ok=True)

        if test_csv_path:
            X_train, y_train = self.read_csv(csv_path)
            X_test, y_test = self.read_csv(test_csv_path)
        else:
            X_train, X_test, y_train, y_test = self.split_train_test(csv_path, test_size=0.6)

        X_train = self.transform(X_train)
        X_test = self.transform(X_test)

        mlp_gs = Perceptron()
        parameter_space = {
            'penalty': ["l2", "l1", "elasticnet", None],
            'alpha': [0.0001, 0.002, 0.005, 0.01, 0.02, 0.07, 0.1, 0.05],
            'l1_ratio': [0.15, 0.3, 0.5, 0.7, 0.9],
            'early_stopping': [True, False]
        }
        k = "c_percep"
        c = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=5)
        c.fit(X_train, y_train)  # X is train samples and y is the corresponding labels
        #print('Best parameters found:\n', c.best_params_)

        path = join(model_folder, f"{self.model_name}_{self.lang}.{k}")

        # calibrate the classifier
        # we want the output to be directly interpretable as a probability

        clf = SklearnOVOSClassifier(self.pipeline_id, c.best_estimator_)
        # clf.train(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = balanced_accuracy_score(y_test, y_pred)

        report = f"Accuracy: {acc}\n" + classification_report(y_test, y_pred, target_names=c.classes_)
       # print(report)
        with open(f'{model_folder}/reports/{self.model_name}_{self.lang}_{k}.txt', "w") as f:
            f.write(report)

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

    def find_best_MLP(self, csv_path, test_csv_path=None, model_folder=None, max_iter=200):
        model_folder = model_folder or f"{dirname(__file__)}/models"
        makedirs(model_folder, exist_ok=True)

        if test_csv_path:
            X_train, y_train = self.read_csv(csv_path)
            X_test, y_test = self.read_csv(test_csv_path)
        else:
            X_train, X_test, y_train, y_test = self.split_train_test(csv_path, test_size=0.6)

        X_train = self.transform(X_train)
        X_test = self.transform(X_test)

        mlp_gs = MLPClassifier(max_iter=max_iter)
        parameter_space = {
            'hidden_layer_sizes': [(random.randint(10, 80), random.randint(80, 150)),
                                   (random.randint(50, 150), random.randint(20, 50)),
                                   (random.randint(20, 150), random.randint(20, 150)),
                                   (random.randint(100, 250),),
                                   (120, 20, 80),
                                   (random.randint(20, 150), random.randint(20, 150), random.randint(20, 150)),
                                   (random.randint(100, 150), random.randint(20, 150), random.randint(20, 50)),
                                   (random.randint(20, 50), random.randint(50, 150), random.randint(20, 150))],
            'activation': ["identity", "logistic", "tanh", "relu"],
            'solver': ['sgd', 'adam', 'lbfgs'],
            'early_stopping': [True, False],
            'alpha': [
                      0.001 * random.randint(1, 10),
                      0.0005 * random.randint(1, 100),
                      0.01 * random.randint(1, 10),
                      0.05],
            'learning_rate': ['constant', 'adaptive', 'invscaling'],
        }
        k = "c_mlp"
        c = RandomizedSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=5)
        c.fit(X_train, y_train)  # X is train samples and y is the corresponding labels
        #print('Best parameters found:\n', c.best_params_)

        # calibrate the classifier
        # we want the output to be directly interpretable as a probability
        calibrate = CalibratedClassifierCV(c.best_estimator_)
        clf = SklearnOVOSClassifier(self.pipeline_id, calibrate)
        clf.enabled_feats = self.clf.enabled_feats
        clf.train(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = balanced_accuracy_score(y_test, y_pred)

        report = f"Balanced Accuracy: {acc}\n" + classification_report(y_test, y_pred, target_names=c.classes_)
        #print(report)
        with open(f'{model_folder}/reports/{self.model_name}_{self.lang}_{k}.txt', "w") as f:
            f.write(report)

        self.clf = clf

        try:
            cm = confusion_matrix(y_test, y_pred, labels=c.classes_)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=c.classes_)
            disp.plot()
            import matplotlib.pyplot as plt
            plt.savefig(f'{model_folder}/reports/{self.model_name}_{k}_cm.png')
        except:
            pass

        return acc, c.best_params_, report

    def save(self, model_folder=None, k="c_mlp"):
        model_folder = model_folder or f"{dirname(__file__)}/models"
        makedirs(model_folder, exist_ok=True)
        path = join(model_folder, f"{self.model_name}_{self.lang}.{k}")
        self.clf.save(path)

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
        elif label == "asmr":
            mt = MediaType.ASMR
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

    def deregister_entity(self, name):
        self.feats.deregister_entity(name)
        self.feats2.deregister_entity(name)

    def transform(self, X):
        if isinstance(X, str):
            X = [X]

        featurizers = []
        if self.feats is not None and \
                "keyword" in self.clf.enabled_feats:
            featurizers.append(self.feats.transform)
        if self.feats2 is not None and \
                "bias" in self.clf.enabled_feats:
            featurizers.append(self.feats2.transform)
        if self.base_clf is not None and \
                "playback_type" in self.clf.enabled_feats:
            featurizers.append(self.base_clf.vectorize)
        if self.base_clf2 is not None and \
                "media_type" in self.clf.enabled_feats:
            featurizers.append(self.base_clf2.vectorize)

        X2 = []
        for f in zip(*(feat(X) for feat in featurizers)):
            f = np.hstack(f)
            X2.append(f)
        return np.array(X2)

    def load(self, model_path=None):
        model_path = model_path or f"{dirname(__file__)}/models/{self.model_name}_{self.lang}.c_mlp"
        super().load(model_path)


class KeywordPlaybackTypeClassifier(BaseKeywordClassifier):
    def __init__(self, lang="all", preload=True, model_name="kword_biased_playback_type",
                 enabled_features=None):
        super().__init__(lang, preload, model_name, enabled_features)

    def load(self, model_path=None):
        model_path = model_path or f"{dirname(__file__)}/models/{self.model_name}_{self.lang}.c_mlp"
        super().load(model_path)


class KeywordBinaryPlaybackClassifier(BaseKeywordClassifier):
    def __init__(self, lang="all", preload=True, model_name="kword_biased_binary_ocp",
                 enabled_features=None):
        super().__init__(lang, preload, model_name, enabled_features)

    def load(self, model_path=None):
        model_path = model_path or f"{dirname(__file__)}/models/{self.model_name}_{self.lang}.c_mlp"
        super().load(model_path)


class KeywordMediaTypeClassifier(BaseKeywordClassifier):
    def __init__(self, base_clf: PlaybackTypeClassifier = None,
                 lang="en", preload=True,
                 model_name="kword_biased_media_type",
                 enabled_features=None):
        super().__init__(lang, preload, model_name, enabled_features)
        if base_clf is None:
            base_clf = KeywordPlaybackTypeClassifier()
            base_clf.load()
        self.base_clf = base_clf
        self.base_clf2 = None

    def load(self, model_path=None):
        model_path = model_path or f"{dirname(__file__)}/models/{self.model_name}_{self.lang}.c_mlp"
        super().load(model_path)


# using keyword feats + other clfs  - lang specific
class BiasedMediaTypeClassifier(KeywordMediaTypeClassifier):
    def __init__(self, base_clf: PlaybackTypeClassifier = None,
                 base_clf2: MediaTypeClassifier = None,
                 lang="en", preload=True,
                 model_name="cv2_biased_media_type",
                 enabled_features=None):
        if base_clf is None:
            base_clf = PlaybackTypeClassifier()
            base_clf.load()
        super().__init__(base_clf, lang, preload, model_name, enabled_features)
        if base_clf2 is None:
            base_clf2 = MediaTypeClassifier()
            base_clf2.load()
        self.base_clf2 = base_clf2

    def load(self, model_path=None):
        model_path = model_path or f"{dirname(__file__)}/models/{self.model_name}_{self.lang}.c_mlp"
        super().load(model_path)


class HeuristicMediaTypeClassifier:
    def __init__(self, lang="en", preload=True):
        self.lang = lang
        self.feats = KeywordFeatures(preload=preload)

    def classification_report(self, csv_path, model_folder=None):
        model_folder = model_folder or f"{dirname(__file__)}/models"

        with open(csv_path) as f:
            lines = f.read().split("\n")[1:]
            lines = [l.split(",", 1) for l in lines if "," in l]

        X_test = [_[1] for _ in lines]
        y_test = [_[0] for _ in lines]

        y_pred = self.predict(X_test)
        acc = balanced_accuracy_score(y_test, y_pred)

        report = f"Balanced Accuracy: {acc}\n" + classification_report(y_test, y_pred,
                                                                       target_names=set(y_test))
        # print(report)
        with open(f'{model_folder}/reports/heuristic_{self.lang}.txt', "w") as f:
            f.write(report)
        # Balanced Accuracy: 0.1794732309758858
        #               precision    recall  f1-score   support
        #
        #         game       1.00      0.00      0.00      1200
        #    audiobook       0.12      0.89      0.21       671
        #      cartoon       0.00      0.00      0.00      1146
        #        anime       0.50      0.65      0.56       961
        #      podcast       0.95      0.91      0.93       779
        #         asmr       0.91      0.72      0.81      1032
        #   tv_channel       0.08      0.91      0.15      1170
        #        music       1.00      0.14      0.24       754
        #           ad       0.94      0.02      0.04       830
        #         news       0.76      0.07      0.12       905
        #  documentary       1.00      0.00      0.01       632
        #       hentai       0.62      0.15      0.24      1005
        #     bw_movie       0.11      0.07      0.09      1194
        # silent_movie       0.00      0.00      0.00       918
        #   short_film       0.06      0.00      0.01      1200
        #  radio_drama       0.07      0.10      0.08      1199
        #          bts       0.00      0.00      0.00       876
        #       series       1.00      0.01      0.02      1196
        #        movie       0.00      0.00      0.00       596
        #   adult_asmr       1.00      0.00      0.01      1147
        #        comic       0.40      0.00      0.00      1176
        #        video       0.00      0.00      0.00       354
        #        radio       0.00      0.00      0.00       939
        #      trailer       1.00      0.00      0.00       504
        #        adult       1.00      0.01      0.02      1172
        #        audio       0.33      0.01      0.01      1093
        #
        #     accuracy                           0.18     24649
        #    macro avg       0.49      0.18      0.14     24649
        # weighted avg       0.49      0.18      0.13     24649
        return report

    def predict(self, X):
        if isinstance(X, str):
            X = [X]
        res = []
        for utt in X:
            ents = self.feats.extract(utt)
            l = "music" # default
            for label, lents in PositiveBias.items():
                if label in ["iot_playback"]:
                    continue
                if any(x in ents for x in lents):
                    l = label
                    break

            res.append(l)
        return res


if __name__ == "__main__":
    ents_csv_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/ocp_entities_v0.csv"
    s_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/ocp_sentences_v0.csv"
    t_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/ocp_playback_type_v0.csv"
    csv_small_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/ocp_media_types_balanced_small_v0.csv"
    csv_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/ocp_media_types_v0.csv"
    csv_big_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/ocp_media_types_balanced_big_v0.csv"

    LOG.set_level("DEBUG")
    # download datasets from https://github.com/NeonJarbas/OCP-dataset

    #
    o = BinaryPlaybackClassifier()  # english only Accuracy 0.99
    # o = BinaryKeywordPlaybackClassifier()  # lang agnostic Accuracy 0.930344394576401
    # o.search_best(s_path)
    o.load()

    preds = o.predict(["play a song", "play my morning jams",
                       "i want to watch the matrix",
                       "tell me a joke", "who are you", "you suck"])
    print(preds)  # ['OCP' 'OCP' 'OCP' 'other' 'other' 'other']

    # basic text only classifier
    clf1 = PlaybackTypeClassifier()
    # clf1.search_best(t_path)  # Accuracy: 0.8139614643381273
    clf1.load()

    # label, confidence = clf1.predict(["play internet radio"], probability=True)[0]
    # print(label, confidence)  # [(audio 0.9668275745715512]
    # label, confidence = clf1.predict(["watch kill bill"], probability=True)[0]
    # print(label, confidence)  # [(video 0.9512718369576729)]

    # basic text only classifier
    clf1 = MediaTypeClassifier("cv2_media_type")
    clf1.search_best(csv_path, test_csv_path=csv_big_path)  # Accuracy: 0.8139614643381273
    # clf1.load()

    label, confidence = clf1.predict(["play metallica"], probability=True)[0]
    print(label, confidence)  # [('music', 0.15532930055019162)]

    exit()
    # keyword biased classifier
    # clf = KeywordMediaTypeClassifier()  # lang agnostic
    # clf.search_best(csv_path, test_csv_path=csv_big_path)
    # clf.load()
    # Accuracy: 0.8868770241199333

    clf = BiasedMediaTypeClassifier(lang="en", preload=True,
                                    model_name="cv2_biased_media_type")  # load entities database
    clf.search_best(csv_path, test_csv_path=csv_big_path)  # Accuracy 0.9311456835977218
    # clf.load()

    # klownevilus is an unknown entity
    labels = clf.predict_labels(["play klownevilus"])[0]
    print(clf.predict(["play klownevilus"])[0])
    print(labels)
    old = labels["movie"]
    old2 = labels["music"]

    # probability increases for movie
    clf.register_entity("movie_name", ["klownevilus"])

    labels2 = clf.predict_labels(["play klownevilus"])[0]
    print(clf.predict(["play klownevilus"])[0])
    n = labels2["movie"]
    n2 = labels2["music"]

    print("bias changed movie confidence: ", old, "to", n)
    print("bias changed music confidence: ", old2, "to", n2)
    # bias changed movie confidence:  0.0495770016040964 to 0.42184841516803073
    # bias changed music confidence:  0.20116106628854769 to 0.0876066334326611
    # assert n > old
    # assert n2 < old2

    from pprint import pprint

    pprint(labels)
    pprint(labels2)
