import dataclasses
import os
import random

from unidecode import unidecode


@dataclasses.dataclass
class UtteranceMatch:
    utterance: str

    # each entry below contains text matched against a keyword database

    # platforms
    audio_steaming_service: str = ""
    video_steaming_service: str = ""

    # publishers
    audiobook_publisher: str = ""
    record_label: str = ""
    film_studio: str = ""

    # genres
    media_type: str = ""
    movie_genre: str = ""
    music_genre: str = ""
    podcast_genre: str = ""
    literary_genre: str = ""

    # known instance names, eg "the matrix"
    book_name: str = ""
    podcast_name: str = ""
    movie_name: str = ""
    cartoon_name: str = ""
    anime_name: str = ""
    tv_channel_name: str = ""
    series_name: str = ""
    radio_name: str = ""
    radio_drama_name: str = ""
    song_name: str = ""
    artist_name: str = ""
    album_name: str = ""
    playlist_name: str = ""


class LookupMatcher:
    def __init__(self, lang, path=None):
        self.lang = lang
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


def generate_samples(p, lang):
    m = LookupMatcher(lang)
    ents = m.load_entities(p)
    templs = m.load_templates(p)

    for media_type, templates in templs.items():
        for t in templates:
            t = t.rstrip(".!?,;:")
            words = t.split()
            slots = [w for w in words if w.startswith("{") and w.endswith("}")]
            if slots and any(s[1:-1] not in ents for s in slots):
                # print(666, t, list(ents.keys()))
                continue
            # t = " ".join([w for w in words if not w.startswith("(")])
            for ent, samples in ents.items():
                #                samples = [s for s in samples if s not in ents["hentai_name"]]
                if ent in t:
                    if not samples:
                        break

                    t = t.replace("{" + ent + "}", random.choice(samples))
            else:
                if "{" not in t:
                    yield media_type, t


if __name__ == "__main__":
    dataset = []

    p = "/home/miro/PycharmProjects/OCP_sprint/ocp-nlp/sparql_ocp"
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
