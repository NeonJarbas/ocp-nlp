from os import makedirs
from os.path import join, dirname

import os
import random
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from unidecode import unidecode

from ovos_classifiers.skovos.tagger import SklearnOVOSClassifier


class MediaTypeClassifier:
    def __init__(self, lang="en", pipeline="cv2"):
        self.lang = lang
        clf = LinearSVC()  # 0.8899733806566105
        # clf = Perceptron() # 0.8598047914818101

        self.pipeline = pipeline
        self.clf = SklearnOVOSClassifier(self.pipeline, clf)

    def train(self, csv_path, model_folder):
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
        path = join(model_folder, f"{self.pipeline}_svc_media_type_{self.lang}.clf")
        self.clf.save(path)
        return acc

    def load(self, model_folder, pipeline=None):
        pipeline = pipeline or self.pipeline
        path = join(model_folder, f"{pipeline}_svc_media_type_{self.lang}.clf")
        self.clf.load_from_file(path)

    def predict(self, utterances):
        return self.clf.predict(utterances)


class BiasedMediaTypeClassifier:
    def __init__(self, lang="en"):
        self.lang = lang
        feats = FeatureUnion([("cv2", CountVectorizer(ngram_range=(1, 2))),
                              ("media", MediaFeaturesVectorizer(lang=self.lang))])

        clf = LinearSVC() # 0.9281277728482697
        # clf2 = ExtraTreesClassifier(n_estimators=10,
        #                            max_depth=None, min_samples_split=2)
        #clf = Perceptron()
        # clf4 = MultinomialNB()
        self.pipeline = "cv2"
        self.clf = SklearnOVOSClassifier(self.pipeline, clf)  # 0.94

        p = Pipeline([
            ("feats", feats),
            ('clf', clf)
        ])

        self.clf = SklearnOVOSClassifier("raw", p)

    def register_entity(self, name, samples):
        self.media_featurizer.register_entity(name, samples)

    def train(self, csv_path, model_folder):
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

        # save pickle
        path = join(model_folder, f"cv2_svc_media_type_biased_{self.lang}.clf")
        self.clf.save(path)

        acc = self.clf.score(X_test, y_test)

        print("Accuracy:", acc)
        # Accuracy:  0.91

        return acc

    @property
    def media_featurizer(self):
        return self.clf._pipeline_clf.steps[0][-1].transformer_list[-1][-1].wordlist

    def load(self, model_folder, pipeline=None):
        pipeline = pipeline or "cv2"
        path = join(model_folder, f"{pipeline}_svc_media_type_biased_{self.lang}.clf")
        self.clf.load_from_file(path)

    def predict(self, utterances):
        return self.clf.predict(utterances)


class WordFeatures:
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

    def __init__(self, lang, path=None, ignore_list=None):
        self.lang = lang
        path = path or f"{dirname(__file__)}/sparql_ocp"
        if ignore_list is None and lang == "en":
            # books/movies etc with this name exist, ignore them
            ignore_list = ["play", "search", "listen", "movie"]
        self.ignore_list = ignore_list or []  # aka stop_words
        if path:
            self.entities = self.load_entities(path)
            self.templates = self.load_templates(path)
        else:
            self.entities = {}
            self.templates = {}

    def register_entity(self, name, samples):
        """ register runtime entity samples,
            eg from skills"""
        if name not in self.entities:
            self.entities[name] = []
        self.entities[name] += samples

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
                ents[e.replace(".intent", "")] = samples

        # from sparql queries - auto generated
        for f in os.listdir(path):
            if not f.endswith(".entity"):
                continue
            # normalize and map to slots
            n = f.replace(".entity", "")

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
        return ents

    def extract(self, sentence, as_bool=False):
        match = {}
        for ent, samples in self.entities.items():
            ent = ent.split("_Q")[0].split(".entity")[0]
            for s in [_ for _ in samples if len(_) > 3 and _.lower() not in self.ignore_list]:
                if s.lower() in sentence.lower():
                    if ent in match:
                        if len(s) > len(match[ent]):
                            match[ent] = s
                    else:
                        match[ent] = s
            if as_bool and ent not in match:
                match[ent] = ""
        if as_bool:
            return {k: str(bool(v)) for k, v in match.items()}
        return match


class MediaFeaturesTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, lang="en", wordlist=None, **kwargs):
        self.lang = lang
        self.wordlist = wordlist or \
                        WordFeatures(self.lang, f"{dirname(__file__)}/sparql_ocp")
        super().__init__(**kwargs)

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        feats = []
        for sent in X:
            s_feature = self.wordlist.extract(sent, as_bool=True)
            feats += [s_feature]
        return feats


class MediaFeaturesVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lang="en", **kwargs):
        super().__init__(**kwargs)
        self.lang = lang
        self.wordlist = WordFeatures(self.lang, f"{dirname(__file__)}/sparql_ocp")
        self._transformer = MediaFeaturesTransformer(lang=lang, wordlist=self.wordlist)
        self._vectorizer = DictVectorizer(sparse=False)

    def get_feature_names(self):
        return self._vectorizer.get_feature_names()

    def fit(self, X, y=None, **kwargs):
        X = self._transformer.transform(X)
        self._vectorizer.fit(X)
        return self

    def transform(self, X, **transform_params):
        X = self._transformer.transform(X, **transform_params)
        return self._vectorizer.transform(X)


if __name__ == "__main__":
    m = MediaFeaturesTransformer()
    # print(m.transform(["play metallica"]))

    model_folder = join(dirname(__file__), "models")
    csv_path = f"{dirname(__file__)}/sparql_ocp/dataset.csv"

    clf = MediaTypeClassifier(lang="en")

    clf.train(csv_path, model_folder)

    clf.load(model_folder)

    print(clf.predict(
        [
            "play metallica",
            "play my morning jams",
            "play a silent movie",
            "play a classic film with zombies",
            "I want to listen to a podcast"
        ]))

    skill_names = ["MySkill", "AwesomeSkill", "AnotherSkill", "BadASSMoviesSkill"]
    movie_names = ["LeMovie", "killer klown", "slow and deadly", "live slow, die old"]

    clf = BiasedMediaTypeClassifier(lang="en")

    #clf.train(csv_path, model_folder)

    clf.load(model_folder)

    print(clf.predict(
        [
            "play metallica",
            "play my morning jams",
            "play a silent movie",
            "play a classic film with zombies",
            "I want to listen to a podcast"
        ]))


    print(clf.clf.predict_proba(["play killer klown", "play slow and deadly"]))


    clf.register_entity("movie_name", movie_names)
    clf.register_entity("movie_streaming_service", skill_names)

    print(clf.clf.predict_proba(["play killer klown", "play slow and deadly"]))

    # ['music' 'music' 'silent' 'movies' 'podcast']

    p = f"{dirname(__file__)}/sparql_ocp"

    l = WordFeatures(lang="en",
                     path=p)

    print(l.extract("play metallica"))
    # {'music_genre': 'Metal', 'artist_name': 'Metallica',
    # 'album_name': 'Metallica', 'game_name': 'METAL', 'movie_name': 'Alli'}

    print(l.extract("play the beatles"))
    # {'series_name': 'The Beatles', 'artist_name': 'The Beatles',
    # 'music_genre': 'Beat', 'album_name': 'The Beatles',
    # 'song_name': 'Play The Beat', 'movie_name': 'The Beatles', 'game_name': 'Beat'}

    print(l.extract("play rob zombie"))
    # {'artist_name': 'Rob Zombie', 'album_name': 'Zombie',
    # 'book_name': 'Zombie', 'game_name': 'Zombie', 'movie_name': 'Zombie'}

    print(l.extract("play horror movie"))
    # {'film_genre': 'Horror', 'cartoon_genre': 'Horror',
    # 'anime_genre': 'Horror', 'video_genre': 'horror',
    # 'book_genre': 'Horror', 'album_name': 'MOVIE',
    # 'book_name': 'Horr', 'movie_name': 'Horror Movie'}

    print(l.extract("play science fiction"))
    #  {'film_genre': 'Science Fiction', 'cartoon_genre': 'Science Fiction',
    #  'podcast_genre': 'Fiction', 'anime_genre': 'Science Fiction',
    #  'documentary_genre': 'Science', 'book_genre': 'Science Fiction',
    #  'artist_name': 'Fiction', 'tv_channel': 'Science',
    #  'album_name': 'Science Fiction', 'short_film_name': 'Science',
    #  'book_name': 'Science Fiction', 'movie_name': 'Science Fiction'}
