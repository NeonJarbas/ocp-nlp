import os.path

import ahocorasick
import numpy as np
import requests
from normality.transliteration import latinize_text
from ovos_config.locations import get_xdg_data_save_path
from ovos_utils.log import LOG
from sklearn.base import BaseEstimator, TransformerMixin

# labels introduce a positive bias without invalidating other labels
PositiveBias = {
    "adult": ["media_type_adult", "pornstar_name", "porn_site", "play_verb_video",
              "porn_genre", "porn_film_name"],
    "adult_asmr": ["media_type_adult_audio", "play_verb_audio"],
    "anime": ["media_type_anime", "anime_name", "play_verb_video"],
    "audio": ["media_type_audio", "audio_genre", "play_verb_audio"],
    "audiobook": ["media_type_audiobook", "book_name",
                  "book_genre", "book_author",
                  "audiobook_streaming_service", "audiobook_narrator", "play_verb_audio"],
    "bts": ["media_type_bts", "play_verb_video"],
    "bw_movie": ["media_type_bw_movie", "bw_movie_name", "play_verb_video"],
    "cartoon": ["media_type_cartoon", "cartoon_name", "play_verb_video"],
    "documentary": ["media_type_documentary", "documentary_name", "play_verb_video"],
    "game": ["gaming_console_name", "game_name", "game_genre"],
    "hentai": ["media_type_hentai", "hentai_name" "hentai_streaming_service", "play_verb_video"],
    "music": ["media_type_music", "music_streaming_service", "album_name", "playlist_name", "soundtrack_keyword",
              "song_name", "record_label", "music_genre", "artist_name", "play_verb_audio"],
    "movie": ["media_type_movie", "movie_streaming_service",
              "movie_name", "film_studio", "film_genre", "play_verb_video"],
    "ad": ["ad_keyword", "movie_name", "play_verb_audio"],
    "news": ["media_type_news", "news_provider", "news_streaming_service", "play_verb_video", "play_verb_audio"],
    "podcast": ["media_type_podcast", "podcast_streaming_service",
                "podcast_name", "podcast_genre", "play_verb_audio"],
    "radio": ["media_type_radio", "radio_streaming_service", "play_verb_audio"],
    "short_film": ["media_type_short_film", "short_film_name", "play_verb_video"],
    "radio_drama": ["media_type_radio_theatre", "radio_drama_name", "play_verb_audio"],
    "silent_movie": ["media_type_silent_movie", "silent_movie_name", "play_verb_video"],
    "trailer": ["media_type_trailer", "play_verb_video"],
    "tv_channel": ["media_type_tv", "tv_streaming_service", "tv_genre", "tv_channel", "play_verb_video"],
    "series": ["media_type_video_episodes", "series_name", "play_verb_video"],
    "comic": ["media_type_visual_story", "comics_genre", "play_verb_video"],
    "video": ["youtube_channel", "video_streaming_service", "play_verb_video"]
}
# labels introduce a negative bias without invalidating the label
NegativeBias = {
    "adult": ["ad_keyword", "play_verb_audio",
              "soundtrack_keyword",
              "music_streaming_service",
              "movie_streaming_service",
              "video_streaming_service",
              "radio_streaming_service",
              "audiobook_streaming_service",
              "podcast_streaming_service",
              "news_streaming_service",

              "media_type_audio",
              "media_type_audiobook", "audiobook_narrator",
              "media_type_bts",
              "media_type_documentary",
              "media_type_music", "album_name", "playlist_name", "song_name", "record_label", "artist_name",
              "media_type_news", "news_provider",
              "media_type_podcast",
              "media_type_radio",
              "media_type_radio_theatre",
              "media_type_trailer",
              ],
    "adult_asmr": ["media_type_music", "music_streaming_service", "album_name", "playlist_name",
                   "song_name", "record_label", "music_genre", "artist_name", "media_type_audiobook", "book_name",
                   "book_genre", "book_author", "media_type_bw_movie", "bw_movie_name", "media_type_radio_theatre",
                   "radio_drama_name", "media_type_music", "music_streaming_service", "album_name", "media_type_audio",
                   "audio_genre", "song_name", "record_label", "music_genre", "artist_name", "movie_streaming_service",
                   "media_type_news", "news_provider", "news_streaming_service", "media_type_podcast",
                   "podcast_streaming_service", "podcast_name", "podcast_genre", "youtube_channel",
                   "video_streaming_service", "media_type_radio", "radio_streaming_service",
                   "audiobook_streaming_service", "media_type_cartoon", "cartoon_name",

                   "soundtrack_keyword", "audiobook_narrator"],
    "anime": ["ad_keyword", "play_verb_audio",
              "soundtrack_keyword",
              "music_streaming_service",
              "radio_streaming_service",
              "audiobook_streaming_service",
              "podcast_streaming_service",
              "news_streaming_service",
              "hentai_streaming_service",
              "porn_site",

              "media_type_adult", "pornstar_name",
              "media_type_adult_audio", "porn_genre", "porn_film_name",

              "media_type_audio",
              "media_type_audiobook", "audiobook_narrator",
              "media_type_bts",
              "media_type_documentary",
              "media_type_music", "album_name", "playlist_name", "song_name", "record_label", "artist_name",
              "media_type_news", "news_provider",
              "media_type_podcast",
              "media_type_radio",
              "media_type_radio_theatre",
              "media_type_trailer",
              ],
    "audio": ["play_verb_video",
              "movie_streaming_service",
              "hentai_streaming_service",
              "porn_site",

              "media_type_adult", "pornstar_name", "porn_site",
              "media_type_adult_audio", "porn_genre", "porn_film_name",
              "media_type_hentai", "hentai_name",

              "media_type_movie", "movie_name",
              "media_type_bw_movie", "bw_movie_name",
              "media_type_silent_movie", "silent_movie_name",
              "media_type_audiobook", "audiobook_narrator",
              "media_type_bts",
              "media_type_documentary",
              "media_type_music", "album_name", "playlist_name", "song_name", "record_label", "artist_name",
              "media_type_podcast",
              "media_type_trailer",
              "media_type_cartoon",
              "media_type_anime"
              ],
    "audiobook": ["play_verb_video",
                  "soundtrack_keyword",
                  "movie_streaming_service",
                  "hentai_streaming_service",
                  "porn_site",

                  "media_type_adult", "pornstar_name", "porn_site",
                  "media_type_adult_audio", "porn_genre", "porn_film_name",
                  "media_type_hentai", "hentai_name",
                  "media_type_movie", "movie_name",
                  "media_type_bw_movie", "bw_movie_name",
                  "media_type_silent_movie", "silent_movie_name",

                  "media_type_bts",
                  "media_type_documentary",
                  "media_type_music", "album_name", "playlist_name", "song_name", "record_label", "artist_name",
                  "media_type_trailer",
                  "media_type_cartoon",
                  "media_type_anime"
                  ],
    "bts": ["ad_keyword", "play_verb_audio",
            "soundtrack_keyword",
            "music_streaming_service",
            "radio_streaming_service",
            "audiobook_streaming_service",
            "podcast_streaming_service",
            "news_streaming_service",
            "hentai_streaming_service",

            "media_type_hentai", "hentai_name",

            "media_type_audio",
            "media_type_audiobook", "audiobook_narrator",
            "media_type_documentary",
            "media_type_music", "album_name", "playlist_name", "song_name", "record_label", "artist_name",
            "media_type_news", "news_provider",
            "media_type_podcast",
            "media_type_radio",
            "media_type_radio_theatre",
            "media_type_trailer",
            ],
    "bw_movie": ["ad_keyword", "play_verb_audio",
                 "soundtrack_keyword",
                 "music_streaming_service",
                 "radio_streaming_service",
                 "audiobook_streaming_service",
                 "podcast_streaming_service",
                 "news_streaming_service",
                 "hentai_streaming_service",
                 "porn_site",

                 "media_type_adult", "pornstar_name", "porn_site",
                 "media_type_adult_audio", "porn_genre", "porn_film_name",
                 "media_type_hentai", "hentai_name",

                 "media_type_audio",
                 "media_type_audiobook", "audiobook_narrator",
                 "media_type_bts",
                 "media_type_documentary",
                 "media_type_music", "album_name", "playlist_name", "song_name", "record_label", "artist_name",
                 "media_type_news", "news_provider",
                 "media_type_podcast",
                 "media_type_radio",
                 "media_type_radio_theatre",
                 "media_type_trailer",
                 "media_type_cartoon",
                 "media_type_anime"
                 ],
    "cartoon": ["ad_keyword", "play_verb_audio",
                "soundtrack_keyword",
                "music_streaming_service",
                "radio_streaming_service",
                "audiobook_streaming_service",
                "podcast_streaming_service",
                "news_streaming_service",
                "hentai_streaming_service",
                "porn_site",

                "media_type_adult", "pornstar_name", "porn_site",
                "media_type_adult_audio", "porn_genre", "porn_film_name",
                "media_type_hentai", "hentai_name",

                "media_type_audio",
                "media_type_audiobook", "audiobook_narrator",
                "media_type_bts",
                "media_type_documentary",
                "media_type_music", "album_name", "playlist_name", "song_name", "record_label", "artist_name",
                "media_type_news", "news_provider",
                "media_type_podcast",
                "media_type_radio",
                "media_type_radio_theatre",
                "media_type_trailer",
                ],
    "documentary": ["ad_keyword", "play_verb_audio",
                    "soundtrack_keyword",
                    "music_streaming_service",
                    "radio_streaming_service",
                    "audiobook_streaming_service",
                    "podcast_streaming_service",
                    "news_streaming_service",
                    "hentai_streaming_service",
                    "porn_site",

                    "media_type_adult", "pornstar_name", "porn_site",
                    "media_type_adult_audio", "porn_genre", "porn_film_name",
                    "media_type_hentai", "hentai_name",

                    "media_type_audio",
                    "media_type_audiobook", "audiobook_narrator",
                    "media_type_bts",
                    "media_type_music", "album_name", "playlist_name", "song_name", "record_label", "artist_name",
                    "media_type_news", "news_provider",
                    "media_type_podcast",
                    "media_type_radio",
                    "media_type_radio_theatre",
                    "media_type_trailer",
                    "media_type_cartoon",
                    "media_type_anime"
                    ],
    "game": ["ad_keyword", "play_verb_audio", "play_verb_video",
             "soundtrack_keyword",
             "music_streaming_service",
             "radio_streaming_service",
             "audiobook_streaming_service",
             "podcast_streaming_service",
             "news_streaming_service",
             "movie_streaming_service",
             "hentai_streaming_service",
             "porn_site",

             "media_type_adult", "pornstar_name", "porn_site",
             "media_type_adult_audio", "porn_genre", "porn_film_name",
             "media_type_hentai", "hentai_name",

             "media_type_audiobook", "audiobook_narrator",
             "media_type_bts",
             "media_type_documentary",
             "media_type_music", "album_name", "playlist_name", "song_name", "record_label", "artist_name",
             "media_type_news", "news_provider",
             "media_type_podcast",
             "media_type_radio",
             "media_type_radio_theatre",
             "media_type_trailer",
             "media_type_cartoon",
             "media_type_anime"
             ],
    "hentai": ["ad_keyword", "play_verb_audio",
               "soundtrack_keyword",
               "movie_streaming_service",
               "video_streaming_service",
               "music_streaming_service",
               "radio_streaming_service",
               "audiobook_streaming_service",
               "podcast_streaming_service",
               "news_streaming_service",

               "media_type_adult", "pornstar_name",
               "media_type_adult_audio", "porn_film_name",

               "media_type_audio",
               "media_type_audiobook", "audiobook_narrator",
               "media_type_bts",
               "media_type_documentary",
               "media_type_music", "album_name", "playlist_name", "song_name", "record_label", "artist_name",
               "media_type_news", "news_provider",
               "media_type_podcast",
               "media_type_radio",
               "media_type_radio_theatre",
               "media_type_trailer"
               ],
    "music": ["play_verb_video",
              "movie_streaming_service",
              "hentai_streaming_service",
              "porn_site",

              "media_type_adult", "pornstar_name",
              "media_type_adult_audio", "porn_genre", "porn_film_name",
              "media_type_hentai", "hentai_name",

              "media_type_movie", "movie_name",
              "media_type_bw_movie", "bw_movie_name",
              "media_type_silent_movie", "silent_movie_name",
              "media_type_audiobook", "audiobook_narrator",
              "media_type_bts",
              "media_type_documentary",
              "media_type_news", "news_provider",
              "media_type_podcast",
              "media_type_radio",
              "media_type_radio_theatre",
              "media_type_trailer",
              "media_type_cartoon",
              "media_type_anime"
              ],
    "movie": ["ad_keyword", "play_verb_audio",
              "soundtrack_keyword",
              "music_streaming_service",
              "radio_streaming_service",
              "audiobook_streaming_service",
              "podcast_streaming_service",
              "news_streaming_service",
              "hentai_streaming_service",
              "porn_site",

              "media_type_adult", "pornstar_name", "porn_site",
              "media_type_adult_audio", "porn_genre", "porn_film_name",
              "media_type_hentai", "hentai_name",

              "media_type_audio",
              "media_type_audiobook", "audiobook_narrator",
              "media_type_bts",
              "media_type_documentary",
              "media_type_music", "album_name", "playlist_name", "song_name", "record_label", "artist_name",
              "media_type_news", "news_provider",
              "media_type_podcast",
              "media_type_radio",
              "media_type_radio_theatre",
              "media_type_trailer",
              "media_type_cartoon",
              "media_type_anime"
              ],
    "ad": [
        "soundtrack_keyword",
        "music_streaming_service",
        "radio_streaming_service",
        "audiobook_streaming_service",
        "podcast_streaming_service",
        "news_streaming_service",
        "hentai_streaming_service",
        "porn_site",

        "media_type_adult", "pornstar_name", "porn_site",
        "media_type_adult_audio", "porn_genre", "porn_film_name",
        "media_type_hentai", "hentai_name",

        "media_type_audio",
        "media_type_audiobook", "audiobook_narrator",
        "media_type_bts",
        "media_type_documentary",
        "media_type_music", "album_name", "playlist_name", "song_name", "record_label", "artist_name",
        "media_type_news", "news_provider",
        "media_type_podcast",
        "media_type_radio",
        "media_type_radio_theatre",
        "media_type_trailer",
        "media_type_cartoon",
        "media_type_anime"
    ],
    "news": [
        "soundtrack_keyword",
        "movie_streaming_service",
        "hentai_streaming_service",
        "porn_site",

        "media_type_adult", "pornstar_name", "porn_site",
        "media_type_adult_audio", "porn_genre", "porn_film_name",
        "media_type_hentai", "hentai_name",

        "media_type_movie", "movie_name",
        "media_type_bw_movie", "bw_movie_name",
        "media_type_silent_movie", "silent_movie_name",
        "media_type_audiobook", "audiobook_narrator",
        "media_type_bts",
        "media_type_music", "album_name", "playlist_name", "song_name", "record_label", "artist_name",
        "media_type_radio_theatre",
        "media_type_trailer",
        "media_type_cartoon",
        "media_type_anime"
    ],
    "podcast": ["play_verb_video",
                "soundtrack_keyword",
                "movie_streaming_service",
                "hentai_streaming_service",
                "porn_site",

                "media_type_adult", "porn_site",
                "media_type_adult_audio", "porn_genre", "porn_film_name",
                "media_type_hentai", "hentai_name",

                "media_type_movie", "movie_name",
                "media_type_bw_movie", "bw_movie_name",
                "media_type_silent_movie", "silent_movie_name",
                "media_type_audiobook", "audiobook_narrator",
                "media_type_bts",
                "media_type_documentary",
                "media_type_music", "album_name", "playlist_name", "song_name", "record_label", "artist_name",
                "media_type_trailer",
                "media_type_cartoon",
                "media_type_anime"
                ],
    "radio": ["play_verb_video",
              "soundtrack_keyword",
              "movie_streaming_service",
              "hentai_streaming_service",
              "porn_site",

              "media_type_adult", "pornstar_name", "porn_site",
              "media_type_adult_audio", "porn_genre", "porn_film_name",
              "media_type_hentai", "hentai_name",

              "media_type_movie", "movie_name",
              "media_type_bw_movie", "bw_movie_name",
              "media_type_silent_movie", "silent_movie_name",
              "media_type_audiobook", "audiobook_narrator",
              "media_type_bts",
              "media_type_documentary",
              "media_type_music", "album_name", "playlist_name", "song_name", "record_label", "artist_name",
              "media_type_podcast",
              "media_type_trailer",
              "media_type_cartoon",
              "media_type_anime"
              ],
    "short_film": ["ad_keyword", "play_verb_audio",
                   "soundtrack_keyword",
                   "music_streaming_service",
                   "radio_streaming_service",
                   "audiobook_streaming_service",
                   "podcast_streaming_service",
                   "news_streaming_service",
                   "hentai_streaming_service",
                   "porn_site",

                   "media_type_adult", "pornstar_name", "porn_site",
                   "media_type_adult_audio", "porn_genre", "porn_film_name",
                   "media_type_hentai", "hentai_name",

                   "media_type_audio",
                   "media_type_audiobook", "audiobook_narrator",
                   "media_type_bts",
                   "media_type_documentary",
                   "media_type_music", "album_name", "playlist_name", "song_name", "record_label", "artist_name",
                   "media_type_news", "news_provider",
                   "media_type_podcast",
                   "media_type_radio",
                   "media_type_radio_theatre",
                   "media_type_trailer",
                   "media_type_cartoon",
                   "media_type_anime"
                   ],
    "radio_drama": ["play_verb_video",
                    "soundtrack_keyword",
                    "movie_streaming_service",
                    "news_streaming_service",
                    "hentai_streaming_service",
                    "porn_site",

                    "media_type_adult", "pornstar_name", "porn_site",
                    "media_type_adult_audio", "porn_genre", "porn_film_name",
                    "media_type_hentai", "hentai_name",

                    "media_type_bts",
                    "media_type_documentary",
                    "media_type_music", "album_name", "playlist_name", "song_name", "record_label", "artist_name",
                    "media_type_news", "news_provider",
                    "media_type_podcast",
                    "media_type_trailer",
                    "media_type_cartoon",
                    "media_type_anime"
                    ],
    "silent_movie": ["ad_keyword", "play_verb_audio",
                     "soundtrack_keyword",
                     "music_streaming_service",
                     "radio_streaming_service",
                     "audiobook_streaming_service",
                     "podcast_streaming_service",
                     "news_streaming_service",
                     "hentai_streaming_service",
                     "porn_site",

                     "media_type_adult", "pornstar_name", "porn_site",
                     "media_type_adult_audio", "porn_genre", "porn_film_name",
                     "media_type_hentai", "hentai_name",

                     "media_type_audio",
                     "media_type_audiobook", "audiobook_narrator",
                     "media_type_bts",
                     "media_type_documentary",
                     "media_type_music", "album_name", "playlist_name", "song_name", "record_label", "artist_name",
                     "media_type_news", "news_provider",
                     "media_type_podcast",
                     "media_type_radio",
                     "media_type_radio_theatre",
                     "media_type_trailer",
                     "media_type_cartoon",
                     "media_type_anime"
                     ],
    "trailer": ["play_verb_audio",
                "soundtrack_keyword",
                "music_streaming_service",
                "radio_streaming_service",
                "audiobook_streaming_service",
                "podcast_streaming_service",
                "news_streaming_service",
                "hentai_streaming_service",
                "porn_site",

                "media_type_adult", "pornstar_name", "porn_site",
                "media_type_adult_audio", "porn_genre", "porn_film_name",
                "media_type_hentai", "hentai_name", "hentai_streaming_service",

                "media_type_audio",
                "media_type_audiobook", "audiobook_narrator",
                "media_type_bts",

                "media_type_music", "album_name", "playlist_name",
                "song_name", "record_label", "artist_name",

                "media_type_news", "news_provider",
                "media_type_podcast",
                "media_type_radio",
                "media_type_radio_theatre", "radio_drama_name",
                "media_type_cartoon",
                ],
    "tv_channel": ["ad_keyword", "play_verb_audio",
                   "soundtrack_keyword",
                   "music_streaming_service",
                   "radio_streaming_service",
                   "audiobook_streaming_service",
                   "podcast_streaming_service",
                   "news_streaming_service",
                   "hentai_streaming_service",
                   "porn_site",

                   "media_type_adult", "pornstar_name", "porn_site",
                   "media_type_adult_audio", "porn_genre", "porn_film_name",
                   "media_type_hentai", "hentai_name",

                   "media_type_audio",
                   "media_type_audiobook", "audiobook_narrator",
                   "media_type_bts",

                   "media_type_music", "album_name", "playlist_name",
                   "song_name", "record_label", "artist_name",

                   "media_type_news", "news_provider",
                   "media_type_podcast",
                   "media_type_radio",
                   "media_type_radio_theatre",
                   "media_type_trailer"
                   ],
    "series": ["ad_keyword", "play_verb_audio",
               "soundtrack_keyword",
               "music_streaming_service",
               "radio_streaming_service",
               "audiobook_streaming_service",
               "podcast_streaming_service",
               "news_streaming_service",
               "hentai_streaming_service",
               "porn_site",

               "media_type_adult", "pornstar_name", "porn_site",
               "media_type_adult_audio", "porn_genre", "porn_film_name",
               "media_type_hentai", "hentai_name",

               "media_type_audio",
               "media_type_audiobook", "audiobook_narrator",
               "media_type_bts",
               "media_type_documentary",

               "media_type_music", "album_name", "playlist_name",
               "song_name", "record_label", "artist_name",

               "media_type_news", "news_provider",
               "media_type_podcast",
               "media_type_radio",
               "media_type_radio_theatre",
               "media_type_trailer"
               ],
    "comic": ["ad_keyword", "play_verb_audio",
              "soundtrack_keyword",
              "music_streaming_service",
              "radio_streaming_service",
              "audiobook_streaming_service",
              "podcast_streaming_service",
              "news_streaming_service",
              "hentai_streaming_service",
              "porn_site",

              "media_type_adult", "pornstar_name", "porn_site",
              "media_type_adult_audio", "porn_genre", "porn_film_name",
              "media_type_hentai", "hentai_name",

              "media_type_audio",
              "media_type_audiobook", "audiobook_narrator",
              "media_type_bts",
              "media_type_documentary",
              "media_type_music", "album_name", "playlist_name", "song_name", "record_label", "artist_name",
              "media_type_news", "news_provider",
              "media_type_podcast",
              "media_type_radio",
              "media_type_radio_theatre",
              "media_type_trailer"
              ],
    "video": ["ad_keyword", "play_verb_audio",
              "soundtrack_keyword",
              "music_streaming_service",
              "radio_streaming_service",
              "audiobook_streaming_service",
              "podcast_streaming_service",
              "news_streaming_service",
              "hentai_streaming_service",
              "porn_site",

              "media_type_adult", "pornstar_name",
              "media_type_adult_audio", "porn_genre", "porn_film_name",
              "media_type_hentai", "hentai_name",

              "media_type_audio",
              "media_type_audiobook", "audiobook_narrator",
              "media_type_bts",
              "media_type_documentary",
              "media_type_music", "album_name", "playlist_name", "song_name", "record_label", "artist_name",
              "media_type_news", "news_provider",
              "media_type_podcast",
              "media_type_radio",
              "media_type_radio_theatre",
              "media_type_trailer",
              "media_type_cartoon",
              "media_type_anime"
              ]
}


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

    def __init__(self, path=None, ignore_list=None, preload=False, debug=True,
                 auto_download=False):
        # auto dl to XDG directory
        if path is None and auto_download:
            os.makedirs(f"/{get_xdg_data_save_path()}/OCP", exist_ok=True)
            path = f"/{get_xdg_data_save_path()}/OCP/ocp_entities_v0.csv"
            if not os.path.isfile(path):
                url = "https://github.com/OpenVoiceOS/ovos-datasets/raw/master/text/ocp_entities_v0.csv"
                r = requests.get(url).text
                with open(path, "w") as f:
                    f.write(r)
                LOG.init(f"downloaded ocp_entities.csv to: {path}")

        # books/movies etc with this name exist, ignore them
        ignore_list = ignore_list or ["play", "search", "listen",
                                      "watch", "view", "start"]

        self.ignore_list = ignore_list
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

    def deregister_entity(self, name):
        """ register runtime entity samples,
            eg from skills"""
        if name in self.entities:
            self.entities.pop(name)
        if name in self.bias:
            self.bias.pop(name)
        if name in self.automatons:
            self.automatons.pop(name)
        if name in self._needs_building:
            self._needs_building.pop(name)

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
                if len(v) <= 3:
                    continue

                if "_name" in k and v.lower() in self.ignore_list:
                    # LOG.debug(f"ignoring {k}:  {v}")
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
            if "_name" in k and v.lower() in self.ignore_list:
                continue
            match[k] += 1
            if v in self.bias.get(k, []):
                # LOG.debug(f"Feature Bias: {k} +1 because of: {v}")
                match[k] += 1
        return match

    def get_bias(self, sentence):
        # print("##", sentence)
        labels = {'video', 'tv_channel', 'movie', 'silent_movie', 'adult_asmr',
                  'cartoon', 'audio', 'anime', 'game', 'bw_movie', 'ad', 'trailer',
                  'radio', 'comic', 'news', 'bts', 'documentary', 'adult', 'hentai',
                  'podcast', 'short_film', 'music', 'radio_drama', 'audiobook', 'series'}
        leftover = sentence.lower()
        match = {l: 0 for l in labels}
        seen = {}
        for k, v in sorted(self.match(sentence),
                           key=lambda k: len(k[1]),
                           reverse=True):
            leftover = leftover.replace(v.lower(), "")
            if k not in seen:
                seen[k] = []
            if any(v in s for s in seen[k]):
                # Skip matches that are a subword of a previous match
                continue
            seen[k].append(v)

            for l, kws in PositiveBias.items():
                if k in kws:
                    match[l] += 1
                    # LOG.debug(f"Bias: {l} +1 because of: {k} {v}")

            for l, kws in NegativeBias.items():
                if k in kws:
                    match[l] -= 2
                    # LOG.debug(f"Bias: {l} -1 because of: {k} {v}")

        # LOG.debug(f"leftover sentence: {leftover}")
        # normalize the matches to contain numbers between 0 and 1
        low = min(v for v in match.values())
        high = max(v for v in match.values())
        if high - low == 0:
            return match
        return {k: round((v - low) / (high - low), 3)
                for k, v in match.items()}

    def extract(self, sentence):
        match = {}
        for k, v in self.match(sentence):
            if k not in match:
                match[k] = v
            elif self.bias.get(k) == v or len(v) > len(match[k]):
                match[k] = v
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
        for x in self._transformer.transform(X):
            feats = []
            for label in self.labels_index:
                if label in x:
                    feats.append(x[label])
                else:
                    feats.append(0)
            X2.append(feats)

        return np.array(X2)


class BiasFeaturesTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, preload=True, dataset_path=None, **kwargs):
        self.wordlist = KeywordFeatures(path=dataset_path,
                                        preload=preload)
        self.labels = ['tv_channel', 'adult_asmr', 'bts', 'radio', 'game', 'video', 'bw_movie', 'trailer', 'comic',
                       'radio_drama', 'documentary', 'movie', 'ad', 'podcast', 'anime', 'cartoon', 'hentai', 'music',
                       'news', 'audio', 'short_film', 'series', 'audiobook', 'silent_movie', 'adult']
        super().__init__(**kwargs)

    def register_entity(self, name, samples):
        """ register runtime entity samples,
            eg from skills"""
        self.wordlist.register_entity(name, samples)

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        if isinstance(X, str):
            X = [X]
        feats = []
        for sent in X:
            s_feature = self.wordlist.get_bias(sent)
            feats += [s_feature]
        return feats


class BiasFeaturesVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, preload=True, dataset_path=None, **kwargs):
        super().__init__(**kwargs)
        self._transformer = BiasFeaturesTransformer(preload=preload, dataset_path=dataset_path)
        # NOTE: changing this list requires retraining the classifier
        self.labels_index = sorted(self._transformer.labels)

    def register_entity(self, name, samples):
        """ register runtime entity samples,
            eg from skills"""
        self._transformer.register_entity(name, samples)

    def fit(self, *args, **kwargs):
        return self

    def transform(self, X, **transform_params):
        X2 = []
        for x in self._transformer.transform(X):
            feats = []
            for label in self.labels_index:
                if label in x:
                    feats.append(x[label])
                else:
                    feats.append(0)
            X2.append(feats)

        return np.array(X2)


if __name__ == "__main__":
    # print(MediaFeaturesTransformer().get_entity_names())
    LOG.set_level("DEBUG")
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
