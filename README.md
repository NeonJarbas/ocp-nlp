to become part of ovos-core pipeline stage

all utterance parsing happens here

# OCP Intents 

## Layer 1 - Unambiguous

Before regular intent stage, taking into account current OCP state  (media ready to play / playing)

Only matches if user unambiguously wants to trigger OCP

uses padacioso for exact matches

- play {query}
- previous  (media needs to be loaded)
- next  (media needs to be loaded)
- pause  (media needs to be loaded)
- play / resume (media needs to be loaded)
- stop (media needs to be loaded)

## Layer 2 - Ambiguous

Using adapt keyword matching for flexibility, runs after high confidence intents

these intents require a play verb + at least one of media_type, media_genre or media_name (from database of registered results)

OCP skills can provide these keywords at runtime, additional keywords for things such as media_genre were collected via SPARQL queries to wikidata

## Layer 3 - Fallback

uses a binary classifier to detect if a query is about media playback

# Media Type Classifier

internally used to tag utterances before OCP search process, this informs the result selection by giving priority to certain skills and helps performance by skipping some skills completely during search

uses a jurebes classifier trained in a large synthetic dataset

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
    VISUAL_STORY = 13  # things like animated comic books
    BEHIND_THE_SCENES = 14
    DOCUMENTARY = 15
    RADIO_THEATRE = 16  # unlike audiobooks usually contain a diverse cast and full audio production
    SHORT_FILM = 17  # typically movies under 45 min
    SILENT_MOVIE = 18
    VIDEO_EPISODES = 19  # tv series etc
    BLACK_WHITE_MOVIE = 20
    CARTOON = 21

    ADULT = 69  # for content filtering # for content filtering
    HENTAI = 70  # for content filtering # for content filtering
```

# Synthetic data

entities have been collected via wikidata SPARQL queries to generate synthetic samples, including language support and regional specific samples

ChatGPT was used to generate sentence templates, entity slots were replaced with wikidata entity values during training