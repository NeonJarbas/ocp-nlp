import numpy as np
import unittest

from ocp_nlp.classify import (MediaTypeClassifier,
                              BinaryPlaybackClassifier,
                              BiasedMediaTypeClassifier,
                              )


class TestClassify(unittest.TestCase):

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
        clf = BiasedMediaTypeClassifier(lang="en", preload=True)
        clf.load()
        self.assertIsNotNone(clf)
        self.assertIsInstance(clf, BiasedMediaTypeClassifier)

        # Test runtime entity influencing prediction
        old_confidence = clf.predict_labels(["play klownevilus"])[0]["movie"]

        clf.register_entity("movie_name", ["klownevilus"])

        new_confidence = clf.predict_labels(["play klownevilus"])[0]["movie"]

        self.assertGreater(new_confidence, old_confidence)

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


if __name__ == '__main__':
    unittest.main()
