**temporary repository** to become part of ovos-core pipeline stage

all utterance parsing happens here

## Datasets

entities have been collected via wikidata SPARQL queries to generate synthetic samples, including language support and regional specific samples

ChatGPT was used to generate sentence templates, entity slots were replaced with wikidata entity values during training

download dataset from https://github.com/NeonJarbas/OCP-dataset

![imagem](https://github.com/NeonJarbas/ocp-nlp/assets/59943014/dc8f670e-8a5c-4ec5-bc40-6f1aea34525d)

## OCP Pipeline

### Layer 1 - Unambiguous

Before regular intent stage, taking into account current OCP state  (media ready to play / playing)

Only matches if user unambiguously wants to trigger OCP

uses padacioso for exact matches

- play {query}
- previous  (media needs to be loaded)
- next  (media needs to be loaded)
- pause  (media needs to be loaded)
- play / resume (media needs to be loaded)
- stop (media needs to be loaded)

```python
from ocp_nlp.intents import OCPPipelineMatcher

ocp = OCPPipelineMatcher()
print(ocp.match_high("play metallica", "en-us"))
# IntentMatch(intent_service='OCP_intents',
#   intent_type='ocp:play',
#   intent_data={'media_type': <MediaType.MUSIC: 2>, 'query': 'metallica',
#                'entities': {'album_name': 'Metallica', 'artist_name': 'Metallica'},
#                'conf': 0.96, 'lang': 'en-us'},
#   skill_id='ovos.common_play', utterance='play metallica')

```

### Layer 2 - Semi-Ambiguous

uses a binary classifier to detect if a query is about media playback

```python
from ocp_nlp.intents import OCPPipelineMatcher

ocp = OCPPipelineMatcher()

print(ocp.match_high("put on some metallica", "en-us"))
# None

print(ocp.match_medium("put on some metallica", "en-us"))
# IntentMatch(intent_service='OCP_media',
#   intent_type='ocp:play',
#   intent_data={'media_type': <MediaType.MUSIC: 2>,
#                'entities': {'album_name': 'Metallica', 'artist_name': 'Metallica', 'movie_name': 'Some'},
#                'query': 'put on some metallica',
#                'conf': 0.9578441098114333},
#   skill_id='ovos.common_play', utterance='put on some metallica')
```

### Layer 3 - Ambiguous

Uses keyword matching and requires at least 1 keyword

OCP skills can provide these keywords at runtime, additional keywords for things such as media_genre were collected via SPARQL queries to wikidata

```python
from ocp_nlp.intents import OCPPipelineMatcher

ocp = OCPPipelineMatcher()

print(ocp.match_medium("i wanna hear metallica", "en-us"))
# None

print(ocp.match_fallback("i wanna hear metallica", "en-us"))
#  IntentMatch(intent_service='OCP_fallback',
#    intent_type='ocp:play',
#    intent_data={'media_type': <MediaType.MUSIC: 2>,
#                 'entities': {'album_name': 'Metallica', 'artist_name': 'Metallica'},
#                 'query': 'i wanna hear metallica',
#                 'conf': 0.5027561091821287},
#    skill_id='ovos.common_play', utterance='i wanna hear metallica')

```

## Classifiers

### Architecture

![imagem](https://github.com/NeonJarbas/ocp-nlp/assets/59943014/8abbd761-221f-4e59-8586-f35db7f48945)

![imagem](https://github.com/NeonJarbas/ocp-nlp/assets/59943014/a52cab17-60e0-4779-9ae9-73c9b4245392)

![imagem](https://github.com/NeonJarbas/ocp-nlp/assets/59943014/2144c6a7-d32d-4b3f-89c3-0151f6257f60)

Efficient entity matching is done via [Aho–Corasick algorithm](https://en.wikipedia.org/wiki/Aho%E2%80%93Corasick_algorithm), keyword features are essentially a keyword count. 

The way the OCP dataset was collected ensures these features were present during training and intrepertable, therefore during runtime any number of entities.csv files can be loaded, OVOS skills can also register their own keywords


### Media Type Classifier

internally used to tag utterances before OCP search process, this informs the result selection by giving priority to certain skills and helps performance by skipping some skills completely during search

uses a scikit-learn classifier trained in a large synthetic dataset

![imagem](https://github.com/NeonJarbas/ocp-nlp/assets/59943014/4cdb440d-4673-4972-8691-4c3c489ff4e8)

```python
class MediaType:
    GENERIC = 0  # nothing else matches
    AUDIO = 1  # things like ambient noises
    MUSIC = 2
    VIDEO = 3  # eg, youtube videos
    AUDIOBOOK = 4
    GAME = 5  # because it shares the verb "play", mostly for disambguation
    PODCAST = 6
    RADIO = 7  # live radio
    NEWS = 8  # news reports
    TV = 9  # live tv stream
    MOVIE = 10
    TRAILER = 11
    AUDIO_DESCRIPTION = 12  # narrated movie for the blind
    VISUAL_STORY = 13  # things like animated comic books
    BEHIND_THE_SCENES = 14
    DOCUMENTARY = 15
    RADIO_THEATRE = 16
    SHORT_FILM = 17  # typically movies under 45 min
    SILENT_MOVIE = 18
    VIDEO_EPISODES = 19  # tv series etc
    BLACK_WHITE_MOVIE = 20
    CARTOON = 21
    ANIME = 22

    ADULT = 69  # for content filtering
    HENTAI = 70  # for content filtering
    ADULT_AUDIO = 71  # for content filtering
```

The features of this classifier have been engineered to allow influencing classifications at runtime based on available skills

Classifier options:
- heuristic based on keyword features (baseline - lang agnostic) ~= 20% accuracy
- trained on text only features (count vectorizer baseline - english) ~= 85% accuracy
- trained on keyword features (lang agnostic - runtime keywords influence classification) ~= 88% accuracy
- trained on probabilities of text only classifier + keyword features (english only - runtime keywords influence classification) ~= 95% accuracy

NOTE: several classification algorithms have been tested, Perceptron and MLP are the most sensitive to the runtime bias properly

### Binary classifier

using the dataset collected for media type + ovos-datasets

![imagem](https://github.com/NeonJarbas/ocp-nlp/assets/59943014/bf9b796e-dc57-4320-a472-5b859b9dfcaf)

Classifier options:
- trained on text only features (count vectorizer baseline - english) ~= 95% accuracy
- trained on keyword features (lang agnostic - runtime keywords influence classification) ~= 90% accuracy


### Usage

check if an utterance is playback related

```python
from ocp_nlp.classify import BinaryPlaybackClassifier

clf = BinaryPlaybackClassifier()
clf.load()
preds = clf.predict(["play a song", "play my morning jams",
                   "i want to watch the matrix",
                   "tell me a joke", "who are you", "you suck"])
print(preds)  # ['OCP' 'OCP' 'OCP' 'other' 'other' 'other']
```

get media type of a playback utterance
```python
from ocp_nlp.classify import MediaTypeClassifier, BiasedMediaTypeClassifier

# basic text only classifier
clf1 = MediaTypeClassifier()
clf1.load()

label, confidence = clf1.predict_prob(["play metallica"])[0]
print(label, confidence)  # [('music', 0.3438956411030462)]

# keyword biased classifier, uses the above internally for extra features
clf = BiasedMediaTypeClassifier(lang="en", preload=True)  # load entities database
clf.load()

# klownevilus is an unknown entity
label, confidence = clf.predict_prob(["play klownevilus"])[0]
print(label, confidence)  # music 0.3398020446925623

# probability increases for movie
clf.register_entity("movie_name", ["klownevilus"])  # movie correctly predicted now
label, confidence = clf.predict_prob(["play klownevilus"])[0]
print(label, confidence)  # movie 0.540225616798516
```

extract keywords based on a wikidata wordlist gathered via SPARQL queries
```python

from ocp_nlp.features import KeywordFeatures

# using feature extractor standalone
l = KeywordFeatures(lang="en")

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

```
