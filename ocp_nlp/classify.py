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
    def __init__(self, lang):
        self.lang = lang

    def load_entities(self):
        pass


def load_entities():
    p = "/home/miro/PycharmProjects/OCP_sprint/ocp-nlp/sparql_ocp/en"
    ents = {
        "playlist_name": ["sing in the shower",
                          "mint acoustic",
                          "piano covers",
                          "rainy day",
                          "mint",
                          "focus beats",
                          "edm bangers",
                          "legendary women of rock",
                          "lo-fi beats",
                          "ultimate indie",
                          "chill hits",
                          "summer party",
                          "classical tranquility",
                          "soundtrack for studying",
                          "love songs",
                          "peaceful piano",
                          "groovy tunes",
                          "80s love songs",
                          "soulful sunday",
                          "coffee shop tunes",
                          "pop punk powerhouses",
                          "all out 00s",
                          "country hits",
                          "coffeehouse blend",
                          "happy hits",
                          "power workout",
                          "rock legends",
                          "funk legends",
                          "electro pop",
                          "dance party",
                          "pop up",
                          "throwback hits",
                          "r&b gems",
                          "diva forever",
                          "soft pop hits",
                          "sing-along indie hits",
                          "classic rock anthems",
                          "karaoke classics",
                          "90s rock anthems",
                          "chill vibes",
                          "timeless love songs",
                          "best of the decade",
                          "acoustic hits",
                          "pop remix",
                          "deep focus",
                          "90s dance hits",
                          "summer throwback",
                          "rock classics",
                          "yacht rock",
                          "all out 80s",
                          "reggae vibes",
                          "all out 90s",
                          "feel-good pop",
                          "pump up the jam",
                          "early pop hits",
                          "songs to sing in the car",
                          "alternative chill",
                          "summer chillout",
                          "country roads",
                          "80s power ballads",
                          "indie coffeehouse",
                          "80s smash hits",
                          "road trip sing-alongs",
                          "the most beautiful songs in the world",
                          "pure moods",
                          "hot country",
                          "rap caviar",
                          "tropical house",
                          "summer road trip",
                          "greatest love songs",
                          "ultimate karaoke",
                          "gold school",
                          "70s disco fever",
                          "jazz classics",
                          "latin heat",
                          "viva latino",
                          "shut up & dance",
                          "classical essentials",
                          "acoustic love",
                          "indie pop bliss",
                          "pop divas",
                          "guilty pleasures",
                          "dance hits",
                          "shower songs",
                          "indie electronic",
                          "ultimate party anthems",
                          "the 50 most loved songs of all time",
                          "today's top hits",
                          "power ballads",
                          "are & be",
                          "indie all-stars",
                          "hot hits uk",
                          "80s new wave",
                          "workout motivation",
                          "90s hip-hop",
                          "piano ballads",
                          "80s hair metal"],
    }
    for f in os.listdir(p):
        if not f.endswith(".entity"):
            continue

        # normalize and map to slots
        n = f.replace(".entity", "")

        if n not in ents:
            ents[n] = []
        with open(f"{p}/{f}") as fi:
            for s in fi.read().split("\n"):
                if s:
                    s = unidecode(s)
                    ents[n].append(s)

    return ents


def load_templates():
    p = "/home/miro/PycharmProjects/OCP_sprint/ocp-nlp/dataset_gen/templates"
    ents = {}
    with open(f"{p}/generic.intent") as f:
        GENERIC = f.read().split("\n")
    for f in os.listdir(p):
        if f == "generic.intent":
            continue
        n = f.replace(".intent", "")
        if n not in ents:
            ents[n] = []
        with open(f"{p}/{f}") as fi:
            for s in fi.read().split("\n"):
                if s.startswith("#") or not s.strip():
                    continue
                ents[n].append(s)
        if n not in ["game"]:
            for g in GENERIC:
                ents[n].append(g.replace("{query}", "{" + n + "_genre}"))
                ents[n].append(g.replace("{query}", "{" + n + "_name}"))
    return ents


def generate_samples():
    ents = load_entities()
    templs = load_templates()

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
    for i in range(3):
        dataset += list(generate_samples())

    with open("../sparql_ocp/dataset.csv", "w") as f:
        f.write("label, sentence\n")
        for label, sentence in dataset:
            print(label, sentence)
            f.write(f"{label}, {sentence}\n")

    # dedup
    r = "/home/miro/PycharmProjects/OCP_sprint/ocp-nlp/dataset_gen/templates"
    for root, folders, files in os.walk(r):
        for f in files:
            with open(f"{root}/{f}") as fi:
                lines = set(fi.read().split("\n"))
            with open(f"{root}/{f}", "w") as fi:
                fi.write("\n".join(sorted(lines)))
