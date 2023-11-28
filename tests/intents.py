import unittest
from unittest.mock import MagicMock, patch
from ocp_nlp.intents import OCP
from ocp_nlp.constants import MediaType
from ovos_bus_client.message import Message


class TestOCP(unittest.TestCase):

    def setUp(self):
        self.bus = MagicMock()
        self.ocp = OCP(self.bus)

    def test_load_resource_files(self):
        self.ocp.load_resource_files()

        self.assertIn('en-us', self.ocp._intents)
        self.assertIn('en-us', self.ocp._dialogs)
        self.assertIn('en-us', self.ocp._vocs)

        intents = ['play.intent', 'music.intent', 'featured.intent', 'movie.intent', 'read.intent', 'pause.intent',
                   'news.intent', 'audio.intent', 'tv.intent', 'podcast.intent', 'hentai.intent', 'game.intent',
                   'next.intent', 'video.intent', 'open.intent', 'resume.intent', 'behind_scenes.intent',
                   'silent_movie.intent', 'documentaries.intent', 'prev.intent', 'short_movie.intent', 'porn.intent',
                   'audiobook.intent', 'bw_movie.intent', 'movietrailer.intent', 'radio.intent', 'radio_drama.intent',
                   'comic.intent']
        for intent in intents:
            self.assertIn(intent, self.ocp._intents['en-us'])

        dialogs = ['setup.hints.dialog', 'just.one.moment.dialog', 'cant.play.dialog', 'play.what.dialog']
        for d in dialogs:
            self.assertIn(d, self.ocp._dialogs['en-us'])

        voc = ['behind_scenes.voc', 'trailer.voc', 'Play.voc', 'Track.voc', 'Pause.voc', 'Next.voc', 'video_only.voc',
               'converse_resume.voc', 'Resume.voc', 'Music.voc', 'audio_only.voc', 'PlayResume.voc', 'Prev.voc']
        for v in voc:
            self.assertIn(v, self.ocp._vocs['en-us'])

        for intent in intents:
            print( intent, self.ocp._intents['en-us'][intent])

    def test_match_utterance_music_intent(self):
        test_cases = [
            ("play some music", MediaType.MUSIC),
            ("start a song", MediaType.MUSIC),
            ("play music", MediaType.MUSIC),
            ("start some song", MediaType.MUSIC),
            # Add more test cases to cover all permutations
        ]

        for utterance, expected_media_type in test_cases:
            with self.subTest(utterance=utterance, expected_media_type=expected_media_type):
                result = self.ocp.classify_media(utterance, "en-us")
                print(utterance, result)
                self.assertEqual(result, expected_media_type)

    def test_match_utterance_podcast_intent(self):
        test_cases = [
            ("play podcast", MediaType.PODCAST),
            ("start podcast", MediaType.PODCAST),
            ("play some podcast", MediaType.PODCAST),
            ("start some podcast", MediaType.PODCAST),
            ("play podcast episode 123", MediaType.PODCAST),
            ("start podcast episode 456", MediaType.PODCAST),
            ("play some podcast episode 789", MediaType.PODCAST),
            # Add more test cases to cover all permutations
        ]

        for utterance, expected_media_type in test_cases:
            with self.subTest(utterance=utterance, expected_media_type=expected_media_type):
                result = self.ocp.classify_media(utterance, "en-us")
                print(utterance, result)
                self.assertEqual(result, expected_media_type)

    def test_match_utterance_radio_intent(self):
        test_cases = [
            ("play radio", MediaType.RADIO),
            ("start a radio station", MediaType.RADIO),
            ("play internet radio", MediaType.RADIO),
            ("start some radio station", MediaType.RADIO),
            ("start some web radio", MediaType.RADIO),
            ("start the radio", MediaType.RADIO),
            ("play the radio station", MediaType.RADIO)
        ]

        for utterance, expected_media_type in test_cases:
            with self.subTest(utterance=utterance, expected_media_type=expected_media_type):
                result = self.ocp.classify_media(utterance, "en-us")
                print(utterance, result)
                self.assertEqual(result, expected_media_type)

    def test_match_utterance_audiobook_intent(self):
        test_cases = [
            ("play some audiobook", MediaType.AUDIOBOOK),
            ("start an audio book", MediaType.AUDIOBOOK),
            ("read a book", MediaType.AUDIOBOOK),
            ("read some audiobook", MediaType.AUDIOBOOK),
            ("play audiobook", MediaType.AUDIOBOOK),  # Add cases for different query variations
            ("start some story", MediaType.AUDIOBOOK),
        ]

        for utterance, expected_media_type in test_cases:
            with self.subTest(utterance=utterance, expected_media_type=expected_media_type):
                result = self.ocp.classify_media(utterance, "en-us")
                print(utterance, result)
                self.assertEqual(result, expected_media_type)

    def test_match_utterance_documentaries_intent(self):
        # test self.ocp.classify_media

        test_cases = [
            ("play some documentaries", MediaType.DOCUMENTARY),
            ("start a documentary", MediaType.DOCUMENTARY),
            ("play documentary", MediaType.DOCUMENTARY),
            ("start some documentaries", MediaType.DOCUMENTARY),
            ("play documentaries", MediaType.DOCUMENTARY),
            ("play some documentaries", MediaType.DOCUMENTARY),
        ]

        for utterance, expected_media_type in test_cases:
            with self.subTest(utterance=utterance, expected_media_type=expected_media_type):
                result = self.ocp.classify_media(utterance, "en-us")
                print(utterance, result)
                self.assertEqual(result, expected_media_type)

    def test_match_utterance_movie_trailers_intent(self):
        # test self.ocp.classify_media

        test_cases = [
            #("play movie trailers", MediaType.TRAILER),
            ("start a film preview", MediaType.TRAILER),
            #("play film trailers", MediaType.TRAILER),
            ("start some movie preview", MediaType.TRAILER),
            ("start a trailer", MediaType.TRAILER),
            ("play a movie preview", MediaType.TRAILER),
            ("start some film preview", MediaType.TRAILER),
        ]

        for utterance, expected_media_type in test_cases:
            with self.subTest(utterance=utterance, expected_media_type=expected_media_type):
                result = self.ocp.classify_media(utterance, "en-us")
                print(utterance, result)
                self.assertEqual(result, expected_media_type)

    def test_match_utterance_movies_intent(self):
        test_cases = [
            ("play a movie", MediaType.MOVIE),
            ("start the film", MediaType.MOVIE),
            ("play movies", MediaType.MOVIE),
            ("start an action film", MediaType.MOVIE)
        ]

        for utterance, expected_media_type in test_cases:
            with self.subTest(utterance=utterance, expected_media_type=expected_media_type):
                result = self.ocp.classify_media(utterance, "en-us")
                print(utterance, result)
                self.assertEqual(result, expected_media_type)

    def test_match_utterance_news_intent(self):
        test_cases = [
            ("play news", MediaType.NEWS),
            ("start a news station", MediaType.NEWS),
            ("play some news", MediaType.NEWS),
            ("start the news", MediaType.NEWS),
            ("start euronews", MediaType.NEWS)
        ]

        for utterance, expected_media_type in test_cases:
            with self.subTest(utterance=utterance, expected_media_type=expected_media_type):
                result = self.ocp.classify_media(utterance, "en-us")
                print(utterance, result)
                self.assertEqual(result, expected_media_type)

    def test_match_utterance_silent_intent(self):
        # test self.ocp.classify_media

        test_cases = [
            ("play silent movie", MediaType.SILENT_MOVIE),
            #("start a silent film", MediaType.SILENT_MOVIE),
            #("play a silent film", MediaType.SILENT_MOVIE),
            ("start silent films", MediaType.SILENT_MOVIE),
            # Add more test cases to cover all permutations
        ]

        for utterance, expected_media_type in test_cases:
            with self.subTest(utterance=utterance, expected_media_type=expected_media_type):
                result = self.ocp.classify_media(utterance, "en-us")
                print(utterance, result)
                self.assertEqual(result, expected_media_type)

    def test_match_utterance_tv_intent(self):
        # test self.ocp.classify_media

        test_cases = [
            ("play TV", MediaType.TV),
            ("start a television show", MediaType.TV),
            ("play network television", MediaType.TV),
            ("start some TV channel", MediaType.TV),
            ("start some TV", MediaType.TV),
            ("start a new TV channel", MediaType.TV),
            ("play the latest episode on TV", MediaType.TV),
            ("start TV channel ESPN", MediaType.TV),
            ("play national geographic on TV", MediaType.TV),
            ("start TV", MediaType.TV),
            ("play a show on television", MediaType.TV),
            ("start TV series", MediaType.TV),
            ("play a television program", MediaType.TV),
            ("start the TV", MediaType.TV),
            ("start a TV network", MediaType.TV)
        ]

        for utterance, expected_media_type in test_cases:
            with self.subTest(utterance=utterance, expected_media_type=expected_media_type):
                result = self.ocp.classify_media(utterance, "en-us")
                print(utterance, result)
                self.assertEqual(result, expected_media_type)

    def test_match_utterance_video_intent(self):
        # test self.ocp.classify_media

        test_cases = [
            ("play video", MediaType.VIDEO),
            ("start a video", MediaType.VIDEO),
            ("play a movie video", MediaType.VIDEO),
            ("start video with special effects", MediaType.VIDEO),
            ("play an action video", MediaType.VIDEO),
            ("start movie trailer video", MediaType.VIDEO),
            ("play a funny video", MediaType.VIDEO),
            ("start video", MediaType.VIDEO),
            ("play animated video", MediaType.VIDEO),
            ("start documentary video", MediaType.VIDEO),
            ("play short film video", MediaType.VIDEO),
            ("start black and white movie video", MediaType.VIDEO),
            ("play cartoon video", MediaType.VIDEO),
            ("start silent movie video", MediaType.VIDEO),
            ("play movie preview video", MediaType.VIDEO),
            ("start a video with visual effects", MediaType.VIDEO),
            ("play behind the scenes video", MediaType.VIDEO),
            ("start a visual story video", MediaType.VIDEO),
            ("play a video", MediaType.VIDEO),
            ("start a video with animation", MediaType.VIDEO),
            ("play video with sound", MediaType.VIDEO),
            ("start a video clip", MediaType.VIDEO),
            ("play video footage", MediaType.VIDEO),
            # Add more test cases to cover all permutations
        ]

        for utterance, expected_media_type in test_cases:
            with self.subTest(utterance=utterance, expected_media_type=expected_media_type):
                result = self.ocp.classify_media(utterance, "en-us")
                print(utterance, result)
                self.assertEqual(result, expected_media_type)

    def test_match_utterance_comic_intent(self):
        # test self.ocp.classify_media

        test_cases = [
            ("play a comic", MediaType.VISUAL_STORY),
            ("start a motion comic", MediaType.VISUAL_STORY),
            ("play visual story", MediaType.VISUAL_STORY),
            ("start a visual comic", MediaType.VISUAL_STORY),
            ("play motion comic", MediaType.VISUAL_STORY),
            ("start an animated comic", MediaType.VISUAL_STORY),
            ("play comic", MediaType.VISUAL_STORY),
            #("start some visual story", MediaType.VISUAL_STORY),
            ("play an animated comic", MediaType.VISUAL_STORY),
            ("start comic", MediaType.VISUAL_STORY),
            ("play a motion comic", MediaType.VISUAL_STORY),
            ("start some visual comic", MediaType.VISUAL_STORY),
            #("play an animated story", MediaType.VISUAL_STORY),
            ("start motion comic", MediaType.VISUAL_STORY),
            ("play visual comic", MediaType.VISUAL_STORY),
            ("start a comic series", MediaType.VISUAL_STORY),
            ("start a visual series", MediaType.VISUAL_STORY),
            ("play some comic", MediaType.VISUAL_STORY),
            ("play visual series", MediaType.VISUAL_STORY),
            #("start comic book", MediaType.VISUAL_STORY),
            #("play motion story", MediaType.VISUAL_STORY),
            ("start animated comic", MediaType.VISUAL_STORY),
            # Add more test cases to cover all permutations
        ]

        for utterance, expected_media_type in test_cases:
            with self.subTest(utterance=utterance, expected_media_type=expected_media_type):
                result = self.ocp.classify_media(utterance, "en-us")
                print(utterance, result)
                self.assertEqual(result, expected_media_type)

    def test_match_utterance_bw_movie_intent(self):
        # test self.ocp.classify_media

        test_cases = [
            ("play black and white movie", MediaType.BLACK_WHITE_MOVIE),
            #("start a black and white film", MediaType.BLACK_WHITE_MOVIE),
            ("play film in black and white", MediaType.BLACK_WHITE_MOVIE),
            #("start some black and white movies", MediaType.BLACK_WHITE_MOVIE),
            ("play a movie in black and white", MediaType.BLACK_WHITE_MOVIE),
            ("start a black and white film series", MediaType.BLACK_WHITE_MOVIE),
            ("play black white movie", MediaType.BLACK_WHITE_MOVIE),
            ("start a movie in black white", MediaType.BLACK_WHITE_MOVIE),
            ("play black and white film", MediaType.BLACK_WHITE_MOVIE),
            #("start a black white film", MediaType.BLACK_WHITE_MOVIE),
            ("play black white movies", MediaType.BLACK_WHITE_MOVIE),
            ("start a black and white cinema", MediaType.BLACK_WHITE_MOVIE),
            ("play a film in black white", MediaType.BLACK_WHITE_MOVIE),
            ("start black and white films", MediaType.BLACK_WHITE_MOVIE),
            #("play a black and white movie", MediaType.BLACK_WHITE_MOVIE),
            #("start a classic black and white film", MediaType.BLACK_WHITE_MOVIE),
            ("play a black and white motion picture", MediaType.BLACK_WHITE_MOVIE),
            ("start a black white cinema", MediaType.BLACK_WHITE_MOVIE),
            # Add more test cases to cover all permutations
        ]

        for utterance, expected_media_type in test_cases:
            with self.subTest(utterance=utterance, expected_media_type=expected_media_type):
                result = self.ocp.classify_media(utterance, "en-us")
                print(utterance, result)
                self.assertEqual(result, expected_media_type)

    def test_match_utterance_bscenes_intent(self):
        # test self.ocp.classify_media

        test_cases = [
            ("play behind the scenes", MediaType.BEHIND_THE_SCENES),
            #("start a behind-the-scenes video", MediaType.BEHIND_THE_SCENES),
            #("play behind-the-scenes footage", MediaType.BEHIND_THE_SCENES),
            #("start some behind-the-scene content", MediaType.BEHIND_THE_SCENES),
            #("play a behind-the-scenes documentary", MediaType.BEHIND_THE_SCENES),
            #("start behind the scenes of the movie", MediaType.BEHIND_THE_SCENES),
            #("play a behind-the-scenes look", MediaType.BEHIND_THE_SCENES),
            #("start behind-the-scenes featurette", MediaType.BEHIND_THE_SCENES),
            ("play behind the scenes of TV show", MediaType.BEHIND_THE_SCENES),
            #("start a behind-the-scene interview", MediaType.BEHIND_THE_SCENES),
            #("play behind-the-scenes material", MediaType.BEHIND_THE_SCENES),
            #("start a behind-the-scenes clip", MediaType.BEHIND_THE_SCENES),
            #("play a video with behind-the-scenes content", MediaType.BEHIND_THE_SCENES),
            ("start behind the scenes footage", MediaType.BEHIND_THE_SCENES),
            #("play behind-the-scenes extras", MediaType.BEHIND_THE_SCENES),
            #("start behind-the-scene feature", MediaType.BEHIND_THE_SCENES),
            #("play the behind-the-scenes of the film", MediaType.BEHIND_THE_SCENES),
            # Add more test cases to cover all permutations
        ]

        for utterance, expected_media_type in test_cases:
            with self.subTest(utterance=utterance, expected_media_type=expected_media_type):
                result = self.ocp.classify_media(utterance, "en-us")
                print(utterance, result)
                self.assertEqual(result, expected_media_type)

    @unittest.skip("TODO - debug")
    def test_match_utterance_hentai_intent(self):
        # test self.ocp.classify_media

        test_cases = [
            ("play hentai", MediaType.HENTAI),
            ("start cartoon porn", MediaType.HENTAI),
            ("play animated porn", MediaType.HENTAI),
            ("start a hentai video", MediaType.HENTAI),
            ("play some hentai", MediaType.HENTAI),
            ("start cartoon porn series", MediaType.HENTAI),
            ("play an animated adult film", MediaType.HENTAI),
            ("start hentai", MediaType.HENTAI),
            ("play explicit cartoon content", MediaType.HENTAI),
            ("start adult animation", MediaType.HENTAI),
            ("play hentai scenes", MediaType.HENTAI),
            ("start animated adult content", MediaType.HENTAI),
            ("play hentai videos", MediaType.HENTAI),
            ("start adult cartoon", MediaType.HENTAI),
            ("play animated explicit material", MediaType.HENTAI),
            ("start hentai movie", MediaType.HENTAI),
            # Add more test cases to cover all permutations
        ]

        for utterance, expected_media_type in test_cases:
            with self.subTest(utterance=utterance, expected_media_type=expected_media_type):
                result = self.ocp.classify_media(utterance, "en-us")
                print(utterance, result)
                self.assertEqual(result, expected_media_type)

    @unittest.skip("TODO - debug")
    def test_match_utterance_porn_intent(self):
        # test self.ocp.classify_media

        test_cases = [
            ("play porn", MediaType.ADULT),
            ("start adult content", MediaType.ADULT),
            ("play explicit material", MediaType.ADULT),
            ("start some adult videos", MediaType.ADULT),
            ("play pornographic content", MediaType.ADULT),
            ("start a porn video", MediaType.ADULT),
            ("play adult films", MediaType.ADULT),
            ("start some explicit material", MediaType.ADULT),
            ("play adult entertainment", MediaType.ADULT),
            ("start a pornographic movie", MediaType.ADULT),
            ("play X-rated content", MediaType.ADULT),
            ("start some porn", MediaType.ADULT),
            ("play adult movies", MediaType.ADULT),
            ("start explicit videos", MediaType.ADULT),
            ("play a mature video", MediaType.ADULT),
            ("start adult films", MediaType.ADULT),
            ("play some X-rated material", MediaType.ADULT),
            ("start adult content with {{query}}", MediaType.ADULT),
            ("play {{query}} adult videos", MediaType.ADULT),
            # Add more test cases to cover all permutations
        ]

        for utterance, expected_media_type in test_cases:
            with self.subTest(utterance=utterance, expected_media_type=expected_media_type):
                result = self.ocp.classify_media(utterance, "en-us")
                print(utterance, result)
                self.assertEqual(result, expected_media_type)

    @unittest.skip("TODO - debug")
    def test_match_utterance_short_intent(self):
        # test self.ocp.classify_media

        test_cases = [
            ("play a short movie", MediaType.SHORT_FILM),
            ("start a short film", MediaType.SHORT_FILM),
            ("play film short", MediaType.SHORT_FILM),
            ("start some short movies", MediaType.SHORT_FILM),
            ("play short film", MediaType.SHORT_FILM),  # Different query variations
            ("start a short", MediaType.SHORT_FILM),
        ]

        for utterance, expected_media_type in test_cases:
            with self.subTest(utterance=utterance, expected_media_type=expected_media_type):
                result = self.ocp.classify_media(utterance, "en-us")
                print(utterance, result)
                self.assertEqual(result, expected_media_type)

    @unittest.skip("TODO - debug")
    def test_match_utterance_radio_drama_intent(self):
        # test self.ocp.match_utterance
        test_cases = [
            ("play radio theatre", MediaType.RADIO_THEATRE),
            ("start radio drama", MediaType.RADIO_THEATRE),
            ("play some radio theatre", MediaType.RADIO_THEATRE),
            ("start a radio drama", MediaType.RADIO_THEATRE)
        ]

        for utterance, expected_media_type in test_cases:
            with self.subTest(utterance=utterance, expected_media_type=expected_media_type):
                result = self.ocp.classify_media(utterance, "en-us")
                print(utterance, result)
                self.assertEqual(result, expected_media_type)


if __name__ == '__main__':
    unittest.main()
