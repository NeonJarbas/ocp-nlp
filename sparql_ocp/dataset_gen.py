# pip install sparqlwrapper
# https://rdflib.github.io/sparqlwrapper/
import os.path
import sys

import regex
from SPARQLWrapper import SPARQLWrapper, JSON
from unidecode import unidecode

endpoint_url = "https://query.wikidata.org/sparql"

LANGS = ["ca", "pt", "es", "fr", "en", "de", "uk", "gl", "ru"]

import random

MANUAL_DEFINITIONS = {  # TODO - find a good wikidata query for these
    "news_genre": ["technology",
                   "international",
                   "business",
                   "health",
                   "local",
                   "politics",
                   "entertainment",
                   "general",
                   "science",
                   "sports",
                   "weather",
                   "breaking"]
}

ENTITIES = {
    # entity examples
    "series_name": {
        "P31": ["Q5398426"]
    },
    "game_name": {
        "P31": ["Q7889"]
    },
    "cartoon_name": {
        "P31": ["Q202866",  # animated film
                "Q17175676",  # animated cartoon
                ]
    },
    "anime_name": {
        "P31": ["Q1107",  # anime
                "Q220898",  # OVA   - TODO filter hentai
                "Q1047299"
                ]
    },
    "tv_channel": {
        "P31": ["Q2001305"]
    },
    "album_name": {
        "P31": ["Q482994"]
    },
    "song_name": {
        "P31": ["Q7302866"]
    },
    "book_name": {
        "P31": ["Q7725634"]
    },
    "radio_drama_name": {
        "P31": ["Q2635894",  # radio theatre
                "Q3511312"  # radio drama
                ]
    },
    "artist_name": {
        "P31": ["Q215380",  # bands
                "Q56816954",  # metal bands
                "Q2491498",  # pop band
                "Q5741069",  # rock band
                "Q2596245",  # jazz band
                "Q7628270",  # studio band
                "Q19464263",  # hip hop group
                "Q109940",  # duet
                ],
        "P106": [
            "Q639669",  # musicians
            "Q2252262",  # rapper
            "Q177220",  # singer
            "Q488205"  # singer-songwriter
        ]
    },
    "radio_program_name": {
        "P31": ["Q226730"]
    },
    "movie_name": {
        "P31": ["Q11424",
                "Q24869",
                "Q7510990",
                "Q200092",
                "Q848512"  # sound film / opposite of silent
                ]
    },
    "silent_movie_name": {
        "P31": ["Q226730"]
    },
    "bw_movie_name": {
        "P31": ["Q2254548"]
    },
    "short_film_name": {
        "P31": ["Q24862"]
    },
    "podcast_name": {
        "P31": ["Q24634210"]
    },
    "documentary_name": {
        "P136": ["Q4164344"],
        "P31": ["Q93204"]
    },

    # content creators
    "news_provider": {
        "P136": ["Q106698131", "Q74303978"],
        "P31": ["Q115477322", "Q1358344", "Q1193236"]
    },
    "record_label": {
        "P31": ["Q18127"]
    },
    "film_studio": {
        "P31": ["Q375336", "Q368290"]
    },
    "youtube_channel": {
        "P31": ["Q17558136"]
    },

    # streaming services
    "audiobook_streaming_service": {
        "P31": ["Q1644277"]  # audiobook_publisher
    },
    "movie_streaming_service": {
        "P31": [
            "Q109509795",  # web broadcaster  - movie
        ]
    },
    "generic_streaming_service": {
        "P31": [
            "Q212805"  # digital library
        ]
    },
    "tv_streaming_service": {
        "P31": [
            "Q10689397",  # television production company
        ]
    },
    "video_streaming_service": {
        "P31": [
            "Q63241860", "Q122759350"
        ]
    },
    "music_streaming_service": {
        "P31": [
            "Q15590336"
        ]
    },
    "radio_streaming_service": {
        "P31": [
            "Q184973"
        ]
    },
    "podcast_streaming_service": {
        "P31": [
            "Q24579448", "Q24581379"
        ]
    },

    # genres
    "podcast_genre": {
        "P31": ["Q104822033"]
    },
    "music_genre": {
        "P31": ["Q188451"]
    },
    "film_genre": {
        "P31": ["Q201658"]
    },
    "book_genre": {
        "P31": ["Q223393"]
    },
    "radio_drama_genre": {
        "P31": ["Q2933978"]
    },
    "tv_genre": {
        "P31": ["Q15961987"]
    },
    "audio_genre": {
        "P31": ["Q108676140"]
    },
    "comics_genre": {
        "P31": ["Q20087698"]
    },
    "game_genre": {
        "P31": ["Q659563"]
    },
    "documentary_genre": {
        "P31": ["Q108466143"]
    }
}
ADULT = {
    # adult content filtering
    "pornstar_name": {
        "P106": ["Q488111", "Q66382950"]
    },
    "porn_genre": {
        "P31": ["Q49148153"]
    },
    "porn_film_name": {
        "P31": ["Q185529", "Q18956797", "Q62015757"]
    },
    "porn_site": {
        "P31": ["Q110643339"]
    },
    "hentai_name": {
        "P136": ["Q172067",
                 "Q3249257",  # adult animation
                 ]
    }
}

# for each language, perform a country specific query
COUNTRIES = {
    "ru": ["Q159"],
    "ca": ["Q5705"],
    "uk": ["Q212"],
    "gl": ["Q3908"],
    "es": [
        "Q29",  # Spain
        "Q96",  # Mexico
        "Q414"  # Argentina
    ],
    "de": [
        "Q183"  # Germany
    ],
    "fr": [
        "Q142"  # France
    ],
    "en": [
        "Q30",  # USA
        "Q145"  # U.K.
    ],
    "pt": [
        "Q45",  # Portugal
        "Q155"  # Brasil
    ]
}

ENTITIES_PER_COUNTRY = ENTITIES.keys()  # filter here if needed


def get_results(endpoint_url, query):
    user_agent = "OCP Python/%s.%s" % (sys.version_info[0], sys.version_info[1])
    sparql = SPARQLWrapper(endpoint_url, agent=user_agent)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    return sparql.query().convert()


for lang in LANGS:
    base = f"{os.path.dirname(__file__)}/{lang}"
    os.makedirs(base, exist_ok=True)

    a = list(ENTITIES.items())
    random.shuffle(a)
    for ent, queries in a:
        if os.path.isfile(f"{base}/{ent}.entity"):
            continue
        titles = []
        for rel, subjs in queries.items():
            random.shuffle(subjs)
            for s in subjs:
                q = """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                  PREFIX wd: <http://www.wikidata.org/entity/>
                  PREFIX wdt: <http://www.wikidata.org/prop/direct/>
                  SELECT ?q ?l
                  WHERE 
                  {""" + \
                    f"?q wdt:{rel} wd:{s} ." + "\nOPTIONAL {" + \
                    f'?q rdfs:label ?l  filter (lang(?l) = "{lang}").' + \
                    """}
                    } 
                    ORDER BY ?l"""

                try:
                    results = get_results(endpoint_url, q)
                except:  # sometimes we get a timeout if the query is too large
                    print("query too big!!", rel, s, lang)
                    continue
                for result in results["results"]["bindings"]:
                    if "l" not in result:
                        continue
                    uid = result["q"]["value"]
                    name = result["l"]["value"]
                    if not regex.sub(r'[^\p{Latin}]', u'', name).strip():
                        continue  # filter non latin alphabet strings
                    name = unidecode(name)
                    print(ent, name)
                    titles.append(name)

        if len(titles):
            with open(f"{base}/{ent}.entity", "w") as f:
                f.write("\n".join(set(titles)))

    for country in COUNTRIES[lang]:
        for ent, queries in ENTITIES.items():
            if os.path.isfile(f"{base}/{ent}_{country}.entity") or \
                    ent not in ENTITIES_PER_COUNTRY:
                continue
            titles = []
            for rel, subjs in queries.items():
                for s in subjs:
                    q = """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
                      PREFIX wd: <http://www.wikidata.org/entity/>
                      PREFIX wdt: <http://www.wikidata.org/prop/direct/>
                      SELECT ?q ?l
                      WHERE 
                      {""" + \
                        f"?q wdt:{rel} wd:{s}; wdt:P495 wd:{country} ." + "\nOPTIONAL {" + \
                        f'?q rdfs:label ?l .' + \
                        """}
                        } 
                        ORDER BY ?l"""

                    results = get_results(endpoint_url, q)
                    for result in results["results"]["bindings"]:
                        if "l" not in result:
                            continue
                        uid = result["q"]["value"]
                        name = result["l"]["value"]
                        if not regex.sub(r'[^\p{Latin}]', u'', name).strip():
                            continue  # filter non latin alphabet strings
                        print(ent, name)
                        titles.append(name)

            if len(titles):
                with open(f"{base}/{ent}_{country}.entity", "w") as f:
                    f.write("\n".join(set(titles)))
