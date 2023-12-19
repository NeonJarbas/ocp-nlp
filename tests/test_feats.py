import unittest

from ocp_nlp.features import KeywordFeatures, MediaFeaturesTransformer, BiasFeaturesTransformer


class TestKeywordFeatures(unittest.TestCase):

    def test_register_entity(self):
        # Test the registration of entities
        keyword_features = KeywordFeatures()
        keyword_features.register_entity("test_entity", ["sample1", "sample2"])
        self.assertIn("test_entity", keyword_features.entities)
        self.assertIn("test_entity", keyword_features.bias)
        self.assertIn("test_entity", keyword_features.automatons)
        self.assertIn("sample1", keyword_features.automatons["test_entity"])

    @unittest.skip("TODO create test_data.csv")
    def test_load_entities(self):
        # Test loading entities from a CSV path
        keyword_features = KeywordFeatures()
        entities = keyword_features.load_entities("test_data.csv")
        self.assertIsInstance(entities, dict)
        self.assertIn("season_number", entities)
        self.assertIn("episode_number", entities)

    def test_extract(self):
        # Test the extract method for keyword features
        keyword_features = KeywordFeatures(preload=True)

        result_metallica = keyword_features.extract("play metallica")
        self.assertEqual(result_metallica, {'album_name': 'Metallica', 'artist_name': 'Metallica',
                                            'book_genre': 'play'})

        result_beatles = keyword_features.extract("play the beatles")
        expected_beatles = {'album_name': 'The Beatles', 'series_name': 'The Beatles',
                            'artist_name': 'The Beatles', 'movie_name': 'The Beatles',
                            'book_genre': 'play'}
        self.assertEqual(result_beatles, expected_beatles)

        result_rob_zombie = keyword_features.extract("play rob zombie")
        expected_rob_zombie = {'artist_name': 'Rob Zombie', 'album_name': 'Zombie',
                               'book_name': 'Zombie', 'game_name': 'Zombie', 'movie_name': 'Zombie',
                               'book_genre': 'play'}
        self.assertEqual(result_rob_zombie, expected_rob_zombie)

        result_horror_movie = keyword_features.extract("play horror movie")
        expected_horror_movie = {'album_name': 'Movie',
                                 'anime_genre': 'Horror',
                                 'book_genre': 'Horror',
                                 'cartoon_genre': 'Horror',
                                 'film_genre': 'Horror',
                                 'media_type_movie': 'movie',
                                 'movie_name': 'Horror Movie',
                                 'radio_drama_genre': 'horror',
                                 'video_genre': 'horror'}
        self.assertEqual(result_horror_movie, expected_horror_movie)

        result_science_fiction = keyword_features.extract("play science fiction")
        expected_science_fiction = {'album_name': 'Science Fiction',
                                    'anime_genre': 'Science Fiction',
                                    'artist_name': 'Fiction',
                                    'book_genre': 'Science Fiction',
                                    'book_name': 'Science Fiction',
                                    'cartoon_genre': 'Science Fiction',
                                    'documentary_genre': 'Science',
                                    'film_genre': 'Science Fiction',
                                    'movie_name': 'Science Fiction',
                                    'podcast_genre': 'Science',
                                    'radio_drama_genre': 'science fiction',
                                    'short_film_name': 'Science',
                                    'tv_channel': 'Science'}
        self.assertEqual(result_science_fiction, expected_science_fiction)


class TestMediaFeaturesTransformer(unittest.TestCase):

    def test_register_entity(self):
        # Test the registration of entities in the transformer
        media_transformer = MediaFeaturesTransformer()
        media_transformer.register_entity("test_entity", ["sample1", "sample2"])
        self.assertIn("test_entity", media_transformer.wordlist.entities)

    # Add more tests for other methods in MediaFeaturesTransformer


class TestBiasFeaturesTransformer(unittest.TestCase):

    def test_register_entity(self):
        # Test the registration of entities in the bias transformer
        bias_transformer = BiasFeaturesTransformer()
        bias_transformer.register_entity("test_entity", ["sample1", "sample2"])
        self.assertIn("test_entity", bias_transformer.wordlist.entities)

    # Add more tests for other methods in BiasFeaturesTransformer


if __name__ == '__main__':
    unittest.main()
