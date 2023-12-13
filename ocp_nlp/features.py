import os.path

import ahocorasick
import numpy as np
import requests
from normality.transliteration import latinize_text
from ovos_config.locations import get_xdg_data_save_path
from ovos_utils.log import LOG
from sklearn.base import BaseEstimator, TransformerMixin


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

    def __init__(self, path=None, ignore_list=None, preload=False, debug=True):
        # auto dl to XDG directory
        if path is None:
            os.makedirs(f"/{get_xdg_data_save_path()}/OCP", exist_ok=True)
            path = f"/{get_xdg_data_save_path()}/OCP/ocp_entities_v0.csv"
            if not os.path.isfile(path):
                url = "https://github.com/OpenVoiceOS/ovos-datasets/raw/master/text/ocp_entities_v0.csv"
                r = requests.get(url).text
                with open(path, "w") as f:
                    f.write(r)
                LOG.init(f"downloaded ocp_entities.csv to: {path}")

        if ignore_list:
            # books/movies etc with this name exist, ignore them
            ignore_list = ["play", "search", "listen", "movie"]

        self.ignore_list = ignore_list or []  # aka stop_words
        self.bias = {}  # just for logging
        self.debug = debug
        self.automatons = {}
        self._needs_building = []
        if path and preload:
            self.entities = self.load_entities(path)
        else:
            self.entities = {}

    def register_entity(self, name, samples):
        """ register runtime entity samples,
            eg from skills"""
        if name not in self.entities:
            self.entities[name] = []
        self.entities[name] += samples
        if name not in self.bias:
            self.bias[name] = []
        self.bias[name] += samples

        if name not in self.automatons:
            self.automatons[name] = ahocorasick.Automaton()
        for s in samples:
            self.automatons[name].add_word(s.lower(), s)

        self._needs_building.append(name)

    def load_entities(self, csv_path):
        ents = {
            "season_number": [str(i) for i in range(30)],
            "episode_number": [str(i) for i in range(100)]
        }
        with open(csv_path) as f:
            lines = f.read().split("\n")[1:]
            data = [l.split(",", 1) for l in lines if "," in l]

        for n, s in data:
            if n not in ents:
                ents[n] = []
            s = latinize_text(s)
            ents[n].append(s)

        for k, samples in ents.items():
            self._needs_building.append(k)
            if k not in self.automatons:
                self.automatons[k] = ahocorasick.Automaton()
            for s in samples:
                self.automatons[k].add_word(s.lower(), s)

        return ents

    def match(self, utt):
        for k, automaton in self.automatons.items():
            if k in self._needs_building:
                automaton.make_automaton()

        self._needs_building = []

        utt = utt.lower().strip(".!?,;:")

        for k, automaton in self.automatons.items():
            for idx, v in automaton.iter(utt):
                if v.lower() in self.ignore_list or len(v) <= 3:
                    continue
                # filter partial words
                if " " not in v:
                    if v.lower() not in utt.split(" "):
                        continue
                if v.lower() + " " in utt or utt.endswith(v.lower()):
                    yield k, v

    def count(self, sentence):
        match = {k: 0 for k in self.entities.keys()}
        for k, v in self.match(sentence):
            match[k] += 1
            if v in self.bias.get(k, []):
                LOG.debug(f"Feature Bias: {k} +1 because of: {v}")
                match[k] += 1
        return match

    def extract(self, sentence, as_bool=False):
        match = {}
        for k, v in self.match(sentence):
            if k not in match:
                match[k] = v
            elif self.bias.get(k) == v or len(v) > len(match[k]):
                match[k] = v

        if as_bool:
            return {k: bool(v) for k, v in match.items()}

        return match


class MediaFeaturesTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, preload=True, dataset_path=None, **kwargs):
        self.wordlist = KeywordFeatures(path=dataset_path,
                                        preload=preload)
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
            s_feature = self.wordlist.count(sent)
            feats += [s_feature]
        return feats


class MediaFeaturesVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, preload=True, dataset_path=None, **kwargs):
        super().__init__(**kwargs)
        self._transformer = MediaFeaturesTransformer(preload=preload, dataset_path=dataset_path)
        # NOTE: changing this list requires retraining the classifier
        self.labels_index = sorted(self._transformer.get_entity_names())

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
                if label in match:
                    feats.append(match[label])
                else:
                    feats.append(0)
            X2.append(feats)

        return np.array(X2)


if __name__ == "__main__":
    print(MediaFeaturesTransformer().get_entity_names())

    # using feature extractor standalone
    l = KeywordFeatures(preload=True)

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
