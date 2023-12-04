from os import makedirs
from os.path import join, dirname

import joblib
import numpy as np
import os
import random
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVC
from unidecode import unidecode

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
        # Accuracy:  0.91

        # save pickle
        path = join(model_folder, f"cv2_svc_media_type_{self.lang}.clf")
        self.clf.save(path)
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
    def __init__(self, base_clf: MediaTypeClassifier, lang="en", preload=False):
        self.lang = lang
        self.base_clf = base_clf
        self.feats = MediaFeaturesVectorizer(lang=self.lang, preload=preload)
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
        # Accuracy:  0.91

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
        # Accuracy:  0.91

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


class KeywordFeatures:
    """ these features introduce a bias to the classification model

    at runtime registered skills can provide keywords that
    explicitly trigger some media_type specific features

    during training a wordlist gathered from wikidata via sparql queries is used to introduce bias

    a biased and an unbiased model are provided, unbiased operates on word features only

    can also be used as is for rudimentary keyword extraction,
        eg. matching genres as auxiliary data for OCP searches

        TODO: new decorator
             @ocp_genre_search
    """

    def __init__(self, lang, path=None, ignore_list=None, preload=False):
        self.lang = lang
        path = path or f"{dirname(__file__)}/sparql_ocp"
        if ignore_list is None and lang == "en":
            # books/movies etc with this name exist, ignore them
            ignore_list = ["play", "search", "listen", "movie"]
        self.ignore_list = ignore_list or []  # aka stop_words
        if path and preload:
            self.entities = self.load_entities(path)
            self.templates = self.load_templates(path)
        else:
            self.entities = {}
            self.templates = {}
        self.bias = {}  # just for logging

    def register_entity(self, name, samples):
        """ register runtime entity samples,
            eg from skills"""
        if name not in self.entities:
            self.entities[name] = []
        self.entities[name] += samples
        if name not in self.bias:
            self.bias[name] = []
        self.bias[name] += samples

    def load_entities(self, path):
        path = f"{path}/{self.lang}"
        ents = {
            "season_number": [str(i) for i in range(20)],
            "episode_number": [str(i) for i in range(50)]
        }

        # non wikidata entity list - manually maintained by users
        for e in os.listdir(f"{path}/dataset_gen"):
            with open(f"{path}/dataset_gen/{e}") as f:
                samples = f.read().split("\n")
                ents[e.replace(".entity", "").split("_Q")[0]] = samples

        # from sparql queries - auto generated
        for f in os.listdir(path):
            if not f.endswith(".entity"):
                continue
            # normalize and map to slots
            n = f.replace(".entity", "").split("_Q")[0]

            if n not in ents:
                ents[n] = []
            with open(f"{path}/{f}") as fi:
                for s in fi.read().split("\n"):
                    if s:
                        s = unidecode(s)
                        ents[n].append(s)

        return ents

    def load_templates(self, path):
        path = f"{path}/{self.lang}/templates"
        ents = {}
        with open(f"{path}/generic.intent") as f:
            GENERIC = f.read().split("\n")
        with open(f"{path}/generic_video.intent") as f:
            GENERIC2 = f.read().split("\n")
        for f in os.listdir(path):
            if f == "generic.intent":
                continue
            n = f.replace(".intent", "")
            if n not in ents:
                ents[n] = []
            with open(f"{path}/{f}") as fi:
                for s in fi.read().split("\n"):
                    if s.startswith("#") or not s.strip():
                        continue
                    ents[n].append(s)
            if n not in ["game"]:
                for g in GENERIC:
                    ents[n].append(g.replace("{query}", "{" + n + "_genre}"))
                    ents[n].append(g.replace("{query}", "{" + n + "_name}"))
            if n in ["movie", "series", "short_film", "silent_movie",
                     "video", "tv_channel", "comic", "bw_movie", "bts",
                     "anime", "cartoon"]:
                for g in GENERIC2:
                    ents[n].append(g.replace("{query}", "{" + n + "_genre}"))
                    ents[n].append(g.replace("{query}", "{" + n + "_name}"))
        return ents

    def extract(self, sentence, as_bool=False):
        match = {}
        for ent, samples in self.entities.items():
            ent = ent.split("_Q")[0].split(".entity")[0]
            for s in [_ for _ in samples if len(_) > 3 and _.lower() not in self.ignore_list]:
                if s.lower() + " " in sentence.lower() or \
                        sentence.lower().strip(".!?,;:").endswith(s.lower()):
                    if ent in match:
                        if len(s) > len(match[ent]):
                            match[ent] = s
                    else:
                        match[ent] = s
            if as_bool and ent not in match:
                match[ent] = ""
        for k, v in match.items():
            if k in self.bias:
                for s in self.bias[k]:
                    if s in sentence:
                        print("BIAS", k, "because of:", s)
        if as_bool:
            return {k: str(bool(v)) for k, v in match.items()}
        return match


class MediaFeaturesTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, lang="en", preload=True, **kwargs):
        self.lang = lang
        self.wordlist = KeywordFeatures(self.lang, f"{dirname(__file__)}/sparql_ocp", preload=preload)
        super().__init__(**kwargs)

    def register_entity(self, name, samples):
        """ register runtime entity samples,
            eg from skills"""
        self.wordlist.register_entity(name, samples)

    def get_entity_names(self):
        return list(self.wordlist.entities.keys())

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        feats = []
        for sent in X:
            s_feature = self.wordlist.extract(sent, as_bool=True)
            feats += [s_feature]
        return feats


class MediaFeaturesVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lang="en", preload=True, **kwargs):
        super().__init__(**kwargs)
        self.lang = lang
        self._transformer = MediaFeaturesTransformer(lang=lang, preload=preload)
        # NOTE: changing this list requires retraining the classifier
        self.labels_index = ['season_number', 'episode_number', 'film_genre', 'cartoon_genre',
                             'news_streaming_service', 'media_type_documentary', 'media_type_adult',
                             'media_type_bw_movie', 'podcast_genre', 'comic_streaming_service', 'music_genre',
                             'media_type_video_episodes', 'anime_genre', 'media_type_audio', 'media_type_bts',
                             'media_type_silent_movie', 'audiobook_streaming_service', 'radio_drama_genre',
                             'media_type_podcast',
                             'radio_theatre_company', 'media_type_short_film', 'media_type_movie', 'news_provider',
                             'documentary_genre', 'radio_theatre_streaming_service', 'podcast_streaming_service',
                             'media_type_tv', 'comic_name', 'media_type_news', 'media_type_music',
                             'media_type_cartoon', 'documentary_streaming_service', 'cartoon_streaming_service',
                             'anime_streaming_service', 'media_type_hentai', 'movie_streaming_service',
                             'media_type_trailer', 'shorts_streaming_service', 'video_genre', 'porn_streaming_service',
                             'playback_device', 'media_type_game', 'playlist_name', 'media_type_video',
                             'media_type_visual_story', 'media_type_radio_theatre', 'media_type_audiobook',
                             'porn_genre', 'book_genre', 'media_type_anime', 'sound', 'media_type_radio', 'album_name',
                             'country_name', 'generic_streaming_service', 'tv_streaming_service', 'radio_drama_name',
                             'film_studio', 'video_streaming_service', 'short_film_name', 'tv_channel',
                             'youtube_channel', 'bw_movie_name', 'radio_drama', 'radio_program_name', 'game_name',
                             'series_name', 'artist_name', 'tv_genre', 'hentai_name', 'podcast_name',
                             'music_streaming_service', 'silent_movie_name', 'book_name', 'gaming_console_name',
                             'record_label', 'radio_streaming_service', 'game_genre', 'anime_name', 'documentary_name',
                             'cartoon_name', 'audio_genre', 'song_name', 'movie_name', 'porn_film_name', 'comics_genre',
                             'radio_program', 'porn_site', 'pornstar_name']

    def register_entity(self, name, samples):
        """ register runtime entity samples,
            eg from skills"""
        self._transformer.register_entity(name, samples)

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        X2 = []
        for match in self._transformer.transform(X):
            feats = []
            for label in self.labels_index:
                if match.get(label) == "True":
                    feats.append(1)
                else:
                    feats.append(0)
            X2.append(feats)

        return np.array(X2)


if __name__ == "__main__":

    # basic text only classifier
    clf1 = MediaTypeClassifier()
    clf1.load()

    label, confidence = clf1.predict_prob(["play metallica"])[0]
    print(label, confidence)  # [('music', 0.3438956411030462)]

    # keyword biased classifier, uses the above internally for extra features
    clf = BiasedMediaTypeClassifier(clf1, lang="en", preload=True)  # load entities database

    # csv_path = f"{dirname(__file__)}/sparql_ocp/dataset.csv"
    #clf.precompute_features(csv_path)
    #clf.train_from_precomputed()

    clf.load()

    # klownevilus is an unknown entity
    label, confidence = clf.predict_prob(["play klownevilus"])[0]
    print(label, confidence)  # music 0.3398020446925623

    # probability increases for movie
    clf.register_entity("movie_name", ["klownevilus"])  # movie correctly predicted now
    label, confidence = clf.predict_prob(["play klownevilus"])[0]
    print(label, confidence)  # movie 0.540225616798516

    # using feature extractor standalone
    l = KeywordFeatures(lang="en", path=f"{dirname(__file__)}/sparql_ocp")

    print(l.extract("play metallica"))
    # {'album_name': 'Metallica', 'artist_name': 'Metallica'}

    print(l.extract("play the beatles"))
    # {'album_name': 'The Beatles', 'series_name': 'The Beatles',
    # 'artist_name': 'The Beatles', 'movie_name': 'The Beatles'}

    print(l.extract("play rob zombie"))
    # {'artist_name': 'Rob Zombie', 'album_name': 'Zombie',
    # 'book_name': 'Zombie', 'game_name': 'Zombie', 'movie_name': 'Zombie'}

    print(l.extract("play horror movie"))
    # {'film_genre': 'Horror', 'cartoon_genre': 'Horror', 'anime_genre': 'Horror',
    # 'radio_drama_genre': 'horror', 'video_genre': 'horror',
    # 'book_genre': 'Horror', 'movie_name': 'Horror Movie'}

    print(l.extract("play science fiction"))
    #  {'film_genre': 'Science Fiction', 'cartoon_genre': 'Science Fiction',
    #  'podcast_genre': 'Fiction', 'anime_genre': 'Science Fiction',
    #  'documentary_genre': 'Science', 'book_genre': 'Science Fiction',
    #  'artist_name': 'Fiction', 'tv_channel': 'Science',
    #  'album_name': 'Science Fiction', 'short_film_name': 'Science',
    #  'book_name': 'Science Fiction', 'movie_name': 'Science Fiction'}

