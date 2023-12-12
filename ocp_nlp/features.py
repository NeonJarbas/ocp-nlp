import os
from os.path import dirname

import ahocorasick
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from unidecode import unidecode


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

    def __init__(self, lang, path=None, ignore_list=None, preload=False, debug=True):
        self.lang = lang
        path = path or f"{dirname(__file__)}/sparql_ocp"
        if ignore_list is None and lang == "en":
            # books/movies etc with this name exist, ignore them
            ignore_list = ["play", "search", "listen", "movie"]
        self.ignore_list = ignore_list or []  # aka stop_words
        self.bias = {}  # just for logging
        self.debug = debug
        self.automatons = {}
        self._needs_building = []
        if path and preload:
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
        if name not in self.bias:
            self.bias[name] = []
        self.bias[name] += samples

        if name not in self.automatons:
            self.automatons[name] = ahocorasick.Automaton()
        for s in samples:
            self.automatons[name].add_word(s.lower(), s)

        self._needs_building.append(name)

    def load_entities(self, path):
        path = f"{path}/{self.lang}"
        ents = {
            "season_number": [str(i) for i in range(30)],
            "episode_number": [str(i) for i in range(100)]
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

        for k, samples in ents.items():
            self._needs_building.append(k)
            if k not in self.automatons:
                self.automatons[k] = ahocorasick.Automaton()
            for s in samples:
                self.automatons[k].add_word(s.lower(), s)

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
            if n not in ["game", "movie", "series", "short_film", "silent_movie",
                         "video", "tv_channel", "comic", "bw_movie", "bts",
                         "anime", "cartoon"]:
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

    def match(self, utt):
        for k, automaton in self.automatons.items():
            if k in self._needs_building:
                automaton.make_automaton()

        self._needs_building = []

        utt = utt.lower().strip(".!?,;:")

        for k, automaton in self.automatons.items():
            for idx, v in automaton.iter(utt):
                if v.lower() in self.ignore_list:
                    continue
                # filter partial words
                if " " not in v:
                    if v.lower() not in utt.split(" "):
                        continue
                if v.lower() + " " in utt or utt.endswith(v.lower()):
                    yield k, v

    def extract(self, sentence, as_bool=False):
        match = {}
        for k, v in self.match(sentence):
            if k not in match:
                match[k] = v
            elif len(v) > len(match[k]):
                match[k] = v

        if self.debug:
            for k, v in match.items():
                if k in self.bias:
                    for s in self.bias[k]:
                        if s in sentence:
                            print("BIAS", k, "because of:", s)

        if as_bool:
            return {k: bool(v) for k, v in match.items()}

        return match


class MediaFeaturesTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, lang="en", preload=True, dataset_path=None, **kwargs):
        self.lang = lang
        self.wordlist = KeywordFeatures(self.lang,
                                        path=dataset_path or f"{dirname(__file__)}/sparql_ocp",
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
            s_feature = self.wordlist.extract(sent, as_bool=True)
            feats += [s_feature]
        return feats


class MediaFeaturesVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lang="en", preload=True, dataset_path=None, **kwargs):
        super().__init__(**kwargs)
        self.lang = lang
        self._transformer = MediaFeaturesTransformer(lang=lang, preload=preload, dataset_path=dataset_path)
        # NOTE: changing this list requires retraining the classifier
        # MediaFeaturesTransformer(dataset_path="...").get_entity_names()
        self.labels_index = ['season_number', 'episode_number', 'film_genre', 'cartoon_genre', 'news_streaming_service',
                             'media_type_documentary', 'media_type_adult', 'media_type_bw_movie', 'podcast_genre',
                             'comic_streaming_service', 'music_genre', 'media_type_video_episodes', 'anime_genre',
                             'media_type_audio', 'media_type_bts', 'media_type_silent_movie',
                             'audiobook_streaming_service', 'radio_drama_genre', 'media_type_podcast',
                             'radio_theatre_company', 'media_type_short_film', 'media_type_movie', 'news_provider',
                             'documentary_genre', 'radio_theatre_streaming_service', 'podcast_streaming_service',
                             'media_type_tv', 'comic_name', 'media_type_adult_audio', 'media_type_news',
                             'media_type_music', 'media_type_cartoon', 'documentary_streaming_service',
                             'cartoon_streaming_service', 'anime_streaming_service', 'media_type_hentai',
                             'movie_streaming_service', 'media_type_trailer', 'shorts_streaming_service', 'video_genre',
                             'porn_streaming_service', 'playback_device', 'media_type_game', 'playlist_name',
                             'media_type_video', 'media_type_visual_story', 'media_type_radio_theatre',
                             'media_type_audiobook', 'porn_genre', 'book_genre', 'media_type_anime', 'sound',
                             'media_type_radio', 'album_name', 'country_name', 'generic_streaming_service',
                             'tv_streaming_service', 'radio_drama_name', 'film_studio', 'video_streaming_service',
                             'short_film_name', 'tv_channel', 'youtube_channel', 'bw_movie_name', 'audiobook_narrator',
                             'radio_drama', 'radio_program_name', 'game_name', 'series_name', 'artist_name', 'tv_genre',
                             'hentai_name', 'podcast_name', 'music_streaming_service', 'silent_movie_name', 'book_name',
                             'gaming_console_name', 'book_author', 'record_label', 'radio_streaming_service',
                             'game_genre', 'anime_name', 'documentary_name', 'cartoon_name', 'audio_genre', 'song_name',
                             'movie_name', 'porn_film_name', 'comics_genre', 'radio_program', 'porn_site',
                             'pornstar_name']

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
                if match.get(label):
                    feats.append(1)
                else:
                    feats.append(0)
            X2.append(feats)

        return np.array(X2)


if __name__ == "__main__":
    # download dataset from https://github.com/NeonJarbas/OCP-dataset
    dataset_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/sparql_ocp"
    print(MediaFeaturesTransformer(dataset_path=dataset_path).get_entity_names())

    # using feature extractor standalone
    l = KeywordFeatures(lang="en", path=dataset_path, preload=True)

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
