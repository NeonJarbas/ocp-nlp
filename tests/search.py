import unittest
import time
from unittest.mock import Mock
from ocp_nlp.constants import MediaType, PlaybackMode
from ocp_nlp.search import OCPQuery
from threading import Lock

from ovos_bus_client.message import Message


class TestOCPQuery(unittest.TestCase):

    def setUp(self):
        self.bus = Mock()
        self.config = {"min_timeout": 5, "max_timeout": 15, "playback_mode": PlaybackMode.AUDIO_ONLY}
        self.query = OCPQuery("test_query", self.bus, media_type=MediaType.GENERIC, config=self.config)

    def test_init(self):
        self.assertEqual(self.query.query, "test_query")
        self.assertEqual(self.query.media_type, MediaType.GENERIC)
        self.assertEqual(self.query.bus, self.bus)
        self.assertEqual(self.query.config, self.config)
        self.assertEqual(self.query.active_skills, {})
        self.assertEqual(self.query.query_replies, [])
        self.assertFalse(self.query.searching)
        self.assertEqual(self.query.search_start, 0)
        self.assertEqual(self.query.query_timeouts, 5)
        self.assertFalse(self.query.has_gui)

    def test_reset(self):
        self.query.active_skills = {"skill_1": Lock(), "skill_2": Lock()}
        self.query.query_replies = [{"results": []}]
        self.query.searching = True
        self.query.search_start = time.time()
        self.query.reset()
        self.assertEqual(self.query.active_skills, {})
        self.assertEqual(self.query.query_replies, [])
        self.assertFalse(self.query.searching)
        self.assertEqual(self.query.search_start, 0)
        self.assertEqual(self.query.query_timeouts, 5)
        self.assertFalse(self.query.has_gui)

    def test_send(self):
        self.query.register_events = Mock()
        self.query.bus.emit = Mock()
        self.query.send()
        self.assertTrue(self.query.searching)
        self.assertGreater(self.query.search_start, 0)
        self.query.register_events.assert_called_once()
        self.query.bus.emit.assert_called_once_with(
            Message("ovos.common_play.query", {"phrase": "test_query", "question_type": MediaType.GENERIC})
        )

    def test_wait(self):
        self.query.searching = True
        self.query.search_start = time.time()
        self.query.remove_events = Mock()
        self.query.wait()
        self.assertFalse(self.query.searching)
        self.query.remove_events.assert_called_once()

        # Test when search exceeds the timeout
        self.query.searching = True
        self.query.search_start = time.time()
        self.query.remove_events = Mock()
        self.query.wait()
        self.assertFalse(self.query.searching)
        self.query.remove_events.assert_called_once()
        # self.assertIn("common play query timeout", log.output[0])

    def test_results(self):
        self.query.query_replies = [
            {"results": [{"title": "Test Result 1"}]},
            {"results": [{"title": "Test Result 2"}]}
        ]
        results = self.query.results
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["results"][0]["title"], "Test Result 1")
        self.assertEqual(results[1]["results"][0]["title"], "Test Result 2")

    def test_register_events(self):
        self.query.bus.on = Mock()
        self.query.register_events()
        self.assertEqual(self.query.bus.on.call_count, 3)

    def test_remove_events(self):
        self.query.bus.remove_all_listeners = Mock()
        self.query.remove_events()
        self.assertEqual(self.query.bus.remove_all_listeners.call_count, 3)

    def test_handle_skill_search_start(self):
        message = Mock(data={"skill_id": "test_skill"})
        self.query.handle_skill_search_start(message)
        self.assertEqual(list(self.query.active_skills.keys())[0], "test_skill")

    def test_handle_skill_response(self):
        self.query.searching = True
        self.query.search_start = time.time()
        message = Mock(
            data={"phrase": "test_query", "timeout": 3, "skill_id": "test_skill", "searching": True, "results": []})
        self.query.handle_skill_response(message)
        self.assertEqual(self.query.query_replies, [])

        message = Mock(data={"phrase": "test_query", "timeout": 3, "skill_id": "test_skill", "searching": False,
                             "results": [{"title": "Test Result"}]})
        self.query.handle_skill_response(message)
        self.assertEqual(len(self.query.query_replies), 1)
        self.assertEqual(self.query.query_replies[0]["results"][0]["title"], "Test Result")
        self.assertTrue(self.query.searching)  # still waiting responses

        # Test handling a search response with a high-confidence match
        self.query.query_replies = []
        self.query.searching = True
        self.query.search_start = time.time()
        message = Mock(data={"phrase": "test_query", "timeout": 3, "skill_id": "test_skill", "searching": False,
                             "results": [{"title": "High Confidence Result", "match_confidence": 90}]})

        self.query.handle_skill_response(message)
        self.assertEqual(len(self.query.query_replies), 1)
        self.assertEqual(self.query.query_replies[0]["results"][0]["title"], "High Confidence Result")
        self.assertFalse(self.query.searching)  # not waiting

    def test_handle_skill_search_end(self):
        message = Mock(data={"skill_id": "test_skill"})
        self.query.active_skills = {"test_skill": Lock()}
        self.query.handle_skill_search_end(message)
        self.assertEqual(self.query.active_skills, {})


if __name__ == '__main__':
    unittest.main()
