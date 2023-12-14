import numpy as np
import unittest

from ocp_nlp.classify import (iter_clfs,
                              MediaTypeClassifier,
                              BinaryPlaybackClassifier,
                              KeywordMediaTypeClassifier,
                              BiasedMediaTypeClassifier,
                              )


class TestClassify(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up common resources for the tests (e.g., paths, data files)
        cls.csv_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/ocp_media_types_v0.csv"
        cls.ents_csv_path = "/home/miro/PycharmProjects/OCP_sprint/OCP-dataset/ocp_entities_v0.csv"

    def test_MediaTypeClassifier(self):
        clf = MediaTypeClassifier()
        self.assertIsNotNone(clf)
        self.assertIsInstance(clf, MediaTypeClassifier)
        clf.load()

    def test_BinaryPlaybackClassifier(self):
        clf = BinaryPlaybackClassifier()
        self.assertIsNotNone(clf)
        self.assertIsInstance(clf, BinaryPlaybackClassifier)
        clf.load()

    def test_BiasedMediaTypeClassifier(self):
        clf = BiasedMediaTypeClassifier(entities_path=self.ents_csv_path)
        clf.load()
        self.assertIsNotNone(clf)
        self.assertIsInstance(clf, BiasedMediaTypeClassifier)

        # Test runtime entity influencing prediction
        self.assertEqual(clf.predict(["play klownevilus"])[0], "music")
        old_confidence = clf.predict_labels(["play klownevilus"])[0]["movie"]
        old_confidence2 = clf.predict_labels(["play klownevilus"])[0]["music"]  # wrong classification

        clf.register_entity("movie_name", ["klownevilus"])

        self.assertEqual(clf.predict(["play klownevilus"])[0], "movie")
        new_confidence = clf.predict_labels(["play klownevilus"])[0]["movie"]  # correct classification
        new_confidence2 = clf.predict_labels(["play klownevilus"])[0]["music"]

        self.assertGreater(new_confidence, old_confidence)
        self.assertLess(new_confidence2, old_confidence2)

    def test_predict(self):
        # This test checks if the prediction methods run successfully without errors
        clf = BinaryPlaybackClassifier()
        clf.load()
        preds = clf.predict(["play a song", "play my morning jams", "i want to watch the matrix"])
        self.assertEqual(len(preds), 3)

        # Make predictions
        utterances = ["play a song", "play my morning jams", "i want to watch the matrix",
                      "tell me a joke", "who are you", "you suck"]
        preds = clf.predict(utterances)

        # Expected predictions
        expected_preds = np.array(['OCP', 'OCP', 'OCP', 'other', 'other', 'other'])

        # Print and assert predictions
        print("Predictions:", preds)
        print("Expected Predictions:", expected_preds)
        np.testing.assert_array_equal(preds, expected_preds)

    def test_transform(self):
        # This test checks if the transform method runs successfully without errors
        clf = BiasedMediaTypeClassifier()
        clf.load()
        X = clf.transform(["play metallica"])
        self.assertIsInstance(X, np.ndarray)

    def test_split_train_test(self):
        # This test checks if the split_train_test method runs successfully without errors
        clf = KeywordMediaTypeClassifier()
        X_train, X_test, y_train, y_test = clf.split_train_test(self.csv_path)
        self.assertIsNotNone(X_train)
        self.assertIsNotNone(X_test)
        self.assertIsNotNone(y_train)
        self.assertIsNotNone(y_test)

    def test_iter_clfs(self):
        # This test checks if the iter_clfs generator runs successfully without errors
        clfs = list(iter_clfs())
        self.assertTrue(len(clfs) > 0)
        self.assertIsInstance(clfs[0], tuple)
        self.assertIsInstance(clfs[0][0], str)
        self.assertIsNotNone(clfs[0][1])


if __name__ == '__main__':
    unittest.main()
