import os.path
from os import makedirs
from os.path import join, dirname

import numpy as np
from ovos_utils.log import LOG
from sklearn.metrics import balanced_accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from ocp_nlp.constants import MediaType
from ocp_nlp.features import MediaFeaturesVectorizer, BiasFeaturesVectorizer, KeywordFeatures
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
        for k, c in iter_clfs(calibrate=True):
            path = join(model_folder, f"{self.model_name}_{self.lang}.{k}")
            if os.path.isfile(path) and not retrain:
                continue

            clf = SklearnOVOSClassifier(self.pipeline_id, c)
            try:
                clf.train(X_train, y_train)
            except:
                continue
            print(c, 'Training completed')

            y_pred = clf.predict(X_test)
            acc = balanced_accuracy_score(y_test, y_pred)

            print(c)
            report = f"Accuracy: {acc}\n" + classification_report(y_test, y_pred, target_names=c.classes_)
            print(report)
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

        print("BEST", best[0].clf, "Accuracy", best[1])
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


class KeywordMediaTypeClassifier(_OCPClassifier):
    def __init__(self, lang="all", preload=True, model_name="kword_biased_media_type", entities_path=None):
        entities_path = entities_path or f"{dirname(__file__)}/models/ocp_entities_v0.csv"
        self.feats = MediaFeaturesVectorizer(preload=preload, dataset_path=entities_path)
        self.feats2 = BiasFeaturesVectorizer(preload=preload, dataset_path=entities_path)
        super().__init__(model_name, lang, pipeline_id="raw")

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

    def load(self, model_path=None):
        model_path = model_path or f"{dirname(__file__)}/models/{self.model_name}_{self.lang}.c_percep"
        super().load(model_path)


class BiasedMediaTypeClassifier(KeywordMediaTypeClassifier):
    def __init__(self, base_clf: MediaTypeClassifier = None, lang="en", preload=True,
                 model_name="cv2_biased_media_type", entities_path=None):
        super().__init__(lang, preload, model_name, entities_path)
        if base_clf is None:
            base_clf = MediaTypeClassifier()
            base_clf.load()
        self.base_clf = base_clf

    def transform(self, X):
        if isinstance(X, str):
            X = [X]
        feats1 = self.feats.transform(X)
        feats2 = self.base_clf.vectorize(X)
        X2 = []
        for f1, f2 in zip(feats1, feats2):
            f = np.hstack((f1, f2))
            X2.append(f)
        return np.array(X2)

    def load(self, model_path=None):
        model_path = model_path or f"{dirname(__file__)}/models/{self.model_name}_{self.lang}.c_mlp"
        super().load(model_path)


class BinaryKeywordPlaybackClassifier(KeywordMediaTypeClassifier):
    def __init__(self, lang="all", preload=True, model_name="kword_biased_binary_ocp", entities_path=None):
        super().__init__(lang, preload, model_name, entities_path)


class HeuristicMediaTypeClassifier:
    def __init__(self, lang="en", preload=True, entities_path=None):
        self.lang = lang
        self.feats = KeywordFeatures(preload=preload, path=entities_path)

    def split_train_test(self, csv_path, test_size=1.0):

        with open(csv_path) as f:
            lines = f.read().split("\n")[1:]
            lines = [l.split(",", 1) for l in lines if "," in l]

        X = [_[1] for _ in lines]
        y = [_[0] for _ in lines]
        return [], X, [], y  # all for testing

    def classification_report(self, X_test, y_test, model_folder=None):
        model_folder = model_folder or f"{dirname(__file__)}/models"
        y_pred = self.predict(X_test)
        acc = balanced_accuracy_score(y_test, y_pred)

        report = f"Balanced Accuracy: {acc}\n" + classification_report(y_test, y_pred,
                                                                       target_names=self.feats.labels)
        # print(report)
        with open(f'{model_folder}/reports/heuristic_{self.lang}.txt', "w") as f:
            f.write(report)
        #  Balanced Accuracy: 0.22182720592334784
        #               precision    recall  f1-score   support
        #
        #       hentai       0.23      0.29      0.25       136
        #  documentary       0.00      0.00      0.00       213
        #          bts       0.07      0.76      0.13        25
        #        comic       0.47      0.47      0.47       117
        #         game       0.29      0.03      0.05       298
        #      podcast       0.10      0.40      0.16       312
        #        anime       0.97      0.44      0.60       232
        # silent_movie       0.00      0.00      0.00       351
        #           ad       0.00      0.00      0.00       486
        #        radio       1.00      0.05      0.10       157
        #   adult_asmr       0.55      0.31      0.40       448
        #   short_film       0.07      0.97      0.14       306
        #  radio_drama       0.73      0.22      0.34       297
        #         news       0.50      0.00      0.01       876
        #        adult       0.17      0.10      0.13      1241
        #    audiobook       0.63      0.45      0.53       284
        #        audio       0.43      0.19      0.26       542
        #        music       0.96      0.09      0.17       246
        #       series       0.82      0.10      0.18       472
        #      cartoon       0.67      0.01      0.03       302
        #      trailer       0.05      0.29      0.09       154
        #        video       0.75      0.33      0.46       403
        #   tv_channel       0.00      0.00      0.00       136
        #        movie       1.00      0.03      0.05        80
        #     bw_movie       0.25      0.00      0.01       446
        #
        #     accuracy                           0.17      8560
        #    macro avg       0.43      0.22      0.18      8560
        # weighted avg       0.40      0.17      0.17      8560
        return report

    def predict_labels(self, X):
        if isinstance(X, str):
            X = [X]
        res = []
        for utt in X:
            bias = self.feats.get_bias(utt)
            ents = self.feats.extract(utt)

            if any(x in ents for x in
                   ['season_number', 'episode_number', 'media_type_video_episodes', 'series_name']):
                bias["series"] = 1.0
            if any(x in ents for x in ['media_type_bw_movie', 'bw_movie_name']):
                bias["bw_movie"] = 1.0
            elif any(x in ents for x in ['media_type_silent_movie', 'silent_movie_name']):
                bias["silent_movie"] = 1.0
            elif any(x in ents for x in ['media_type_short_film', 'short_film_name']):
                bias["short_film"] = 1.0
            elif any(x in ents for x in ['media_type_movie', 'film_studio', 'movie_name']):
                bias["movie"] = 1.0
            if any(x in ents for x in ['media_type_documentary', 'documentary_name']):
                bias["documentary"] = 1.0
            if any(x in ents for x in ['media_type_cartoon', 'cartoon_name']):
                bias["cartoon"] = 1.0
            if any(x in ents for x in ['anime_name', 'media_type_anime']):
                bias["anime"] = 1.0
            if any(x in ents for x in ['media_type_hentai', 'hentai_name']):
                bias = {k: 0 for k in bias}
                bias["hentai"] = 1.0
            if any(x in ents for x in ['media_type_video', 'youtube_channel']):
                bias["video"] = 1.0
            if any(x in ents for x in ['media_type_tv', 'tv_channel']):
                bias["tv_channel"] = 1.0
            if any(x in ents for x in ['pornstar_name', 'media_type_adult', 'porn_genre',
                                       'porn_film_name']):
                bias = {k: 0 for k in bias}
                bias["adult"] = 1.0
            if any(x in ents for x in ['media_type_radio']):
                bias["radio"] = 1.0
            if any(x in ents for x in ['media_type_trailer']):
                bias["trailer"] = 1.0
            if any(x in ents for x in ['media_type_bts']):
                bias["bts"] = 1.0
            if any(x in ents for x in ['comic_name']):
                bias["comic"] = 1.0
            if any(x in ents for x in ['soundtrack_keyword',
                                       'playlist_name',
                                       'album_name',
                                       'artist_name',
                                       'song_name', 'media_type_music',
                                       'record_label']):
                bias["music"] = 1.0
            elif any(x in ents for x in ['book_genre',
                                         'audiobook_narrator',
                                         'book_name', 'media_type_audiobook',
                                         'book_author']):
                bias["audiobook"] = 1.0
            elif any(x in ents for x in ['media_type_podcast', 'podcast_name']):
                bias["podcast"] = 1.0
            elif any(x in ents for x in ['radio_theatre_company', 'media_type_radio_theatre',
                                         'radio_drama_name']):
                bias["radio_drama"] = 1.0
            elif any(x in ents for x in ['audio_genre', 'media_type_audio']):
                bias["audio"] = 1.0
            if any(x in ents for x in ['media_type_adult_audio', 'porn_genre', 'pornstar_name',
                                       'media_type_hentai', 'media_type_adult',
                                       'porn_film_name']):
                bias["adult_asmr"] = 1.0

            if any(x in ents for x in ['media_type_news', 'news_provider', 'news_streaming_service']):
                bias = {k: 0 for k in bias}
                bias["news"] = 1.0
            if any(x in ents for x in ['ad_keyword']):
                bias = {k: 0 for k in bias}
                bias["ad"] = 1.0
            if any(x in ents for x in ['game_name', 'media_type_game',
                                       'gaming_console_name']):
                bias = {k: 0 for k in bias}
                bias["game"] = 1.0
            res.append(bias)
        return res

    def predict(self, X):
        labels = self.predict_labels(X)
        return [max(x, key=lambda k: x[k]) for x in labels]

    def vectorize(self, X):
        labels = self.predict_labels(X)
        return [list(x.values()) for x in labels]


if __name__ == "__main__":
    ents_csv_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/ocp_entities_v0.csv"
    s_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/ocp_sentences_v0.csv"
    csv_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/ocp_media_types_v0.csv"

    LOG.set_level("DEBUG")
    # download datasets from https://github.com/NeonJarbas/OCP-dataset

    o = BinaryPlaybackClassifier()
    # o.search_best(s_path)  # 0.99
    o.load()

    clf = BinaryKeywordPlaybackClassifier()
    clf.search_best(s_path)
    clf.load()

    preds = clf.predict(["play a song", "play my morning jams",
                         "i want to watch the matrix",
                         "tell me a joke", "who are you", "you suck"])
    print(preds)  # ['OCP' 'OCP' 'OCP' 'other' 'other' 'other']

    csv_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/ocp_media_types_v0.csv"

    # basic text only classifier
    clf1 = MediaTypeClassifier()
    # clf1.split_train_test(csv_path)
    # clf1.search_best(csv_path)  # Accuracy: 0.8139614643381273
    # m = "/home/miro/PycharmProjects/OCP_sprint/ocp-nlp/ocp_nlp/models/cv2_media_type_en.c_fs_lsvc_mlp"
    clf1.load()

    label, confidence = clf1.predict(["play metallica"], probability=True)[0]
    print(label, confidence)  # [('music', 0.15532930055019162)]

    # keyword biased classifier, uses the above internally for extra features
    # clf = KeywordMediaTypeClassifier()  # lang agnostic
    # clf.search_best(csv_path)  # Accuracy 0.7925217191210133
    #               precision    recall  f1-score   support
    #
    #           ad       0.75      0.56      0.64        82
    #        adult       0.66      0.71      0.69       128
    #   adult_asmr       0.93      0.93      0.93        15
    #        anime       0.81      0.71      0.76        70
    #        audio       0.79      0.72      0.75       179
    #    audiobook       0.93      0.93      0.93       187
    #          bts       0.90      0.86      0.88       139
    #     bw_movie       0.95      0.89      0.92       211
    #      cartoon       0.85      0.78      0.81       291
    #        comic       0.60      0.74      0.66        94
    #  documentary       0.82      0.84      0.83       269
    #         game       0.93      0.88      0.90       184
    #       hentai       0.89      0.88      0.88       178
    #        movie       0.77      0.86      0.81       525
    #        music       0.84      0.89      0.86       745
    #         news       0.95      0.98      0.97       170
    #      podcast       0.80      0.84      0.82       325
    #        radio       0.77      0.64      0.70       148
    #  radio_drama       0.87      0.88      0.87       283
    #       series       0.78      0.85      0.81       181
    #   short_film       0.56      0.51      0.53        92
    # silent_movie       0.91      0.84      0.88       242
    #      trailer       0.63      0.56      0.59        82
    #   tv_channel       0.91      0.81      0.86        48
    #        video       0.77      0.71      0.74       268
    #
    #     accuracy                           0.82      5136
    #    macro avg       0.81      0.79      0.80      5136
    # weighted avg       0.83      0.82      0.82      5136
    # clf.load()

    clf = BiasedMediaTypeClassifier(lang="en", preload=True,
                                    entities_path=ents_csv_path)  # load entities database
    # clf = KeywordMediaTypeClassifier()
    # clf.search_best(csv_path)  # Accuracy 0.9311456835977218
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
