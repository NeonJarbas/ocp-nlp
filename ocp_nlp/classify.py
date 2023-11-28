from os import makedirs
from os.path import join, dirname

import os
import random
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from unidecode import unidecode

from ovos_classifiers.skovos.tagger import SklearnOVOSVotingClassifierTagger


class MediaTypeClassifier:
    def __init__(self, pipeline="cv2"):
        clf1 = LinearSVC()
        clf2 = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2)
        clf3 = Perceptron()
        clf4 = MultinomialNB()
        estimators = [clf1, clf2, clf3, clf4]
        self.pipeline = pipeline
        self.clf = SklearnOVOSVotingClassifierTagger(estimators, self.pipeline)

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
        path = join(model_folder, f"{self.pipeline}_media_type.clf")
        self.clf.save(path)
        return acc

    def load(self, model_folder, pipeline=None):
        pipeline = pipeline or self.pipeline
        path = join(model_folder, f"{pipeline}_media_type.clf")
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

    def load_entities(self, path):
        path = f"{path}/{self.lang}"
        ents = {
            "episode_number": [str(i) for i in range(50)]
        }

        for e in os.listdir(f"{path}/dataset_gen"):
            with open(f"{path}/dataset_gen/{e}") as f:
                samples = f.read().split("\n")
                ents[e.replace(".intent", "")] = samples

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
            if as_bool:
                match[ent] = ""
            for s in [_ for _ in samples if len(_) > 3 and _.lower() not in self.ignore_list]:
                if s.lower() in sentence.lower():
                    if ent in match:
                        if len(s) > len(match[ent]):
                            match[ent] = s
                    else:
                        match[ent] = s
        if as_bool:
            return {k: bool(v) for k, v in match.items()}
        return match


# dataset generator
def generate_samples(p, lang):
    m = WordFeatures(lang)
    ents = m.load_entities(p)
    templs = m.load_templates(p)

    for media_type, templates in templs.items():
        for t in templates:
            t = t.rstrip(".!?,;:")
            words = t.split()
            slots = [w for w in words if w.startswith("{") and w.endswith("}")]
            if slots and any(s[1:-1] not in ents for s in slots):
                continue
            for ent, samples in ents.items():
                if ent in t:
                    if not samples:
                        break
                    t = t.replace("{" + ent + "}", random.choice(samples))

            if "{" not in t:
                yield media_type, t
            else:
                print("bad template", t)


if __name__ == "__main__":

    model_folder = join(dirname(__file__), "models")
    csv_path = "/home/miro/PycharmProjects/OCP_sprint/ocp-nlp/sparql_ocp/dataset.csv"
    clf = MediaTypeClassifier()

    # clf.train(csv_path, model_folder)

    clf.load(model_folder)

    print(clf.predict(
        [
            "play metallica",
            "play rob zombie",
            "play a silent movie",
            "play a classic film with zombies",
            "I want to listen to a podcast"
        ]))
    # ['music' 'music' 'silent' 'movies' 'podcast']

    p = "/home/miro/PycharmProjects/OCP_sprint/ocp-nlp/sparql_ocp"

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


    dataset = []

    lang = "en"
    for i in range(3):
        dataset += list(generate_samples(p, lang))

    with open("../sparql_ocp/dataset.csv", "w") as f:
        f.write("label, sentence\n")
        for label, sentence in dataset:
            f.write(f"{label}, {sentence}\n")

    # dedup
    r = "/home/miro/PycharmProjects/OCP_sprint/ocp-nlp/sparql_ocp"
    for root, folders, files in os.walk(r):
        for f in files:
            if f.endswith(".py") or f.endswith(".csv"):
                continue
            with open(f"{root}/{f}") as fi:
                lines = set(fi.read().split("\n"))
            with open(f"{root}/{f}", "w") as fi:
                fi.write("\n".join(sorted(lines)))
