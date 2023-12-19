import unittest
from ovos_utils.messagebus import FakeBus, Message

from ocp_nlp.intents import OCPPipelineMatcher, PlayerState
from ovos_core.intent_services import IntentMatch


class TestOCPPipelineMatcher(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.ocp = OCPPipelineMatcher(bus=FakeBus())

    def assertIntentMatch(self, result, intent_service, intent_type):
        self.assertIsInstance(result, IntentMatch)
        self.assertEqual(result.intent_service, intent_service)
        self.assertEqual(result.intent_type, intent_type)
        self.assertIn('media_type', result.intent_data)
        self.assertIn('query', result.intent_data)
        self.assertIn('entities', result.intent_data)
        self.assertIn('conf', result.intent_data)

    def test_match_high(self):
        result = self.ocp.match_high("play metallica", "en-us")
        self.assertIntentMatch(result, 'OCP_intents', 'ocp:play')

        result = self.ocp.match_high("put on some metallica", "en-us")
        self.assertIsNone(result)

    def test_match_medium(self):
        result = self.ocp.match_medium("put on some metallica", "en-us")
        self.assertIntentMatch(result, 'OCP_media', 'ocp:play')

        result = self.ocp.match_medium("i wanna hear metallica", "en-us")
        self.assertIsNone(result)

    def test_match_fallback(self):
        result = self.ocp.match_fallback("i wanna hear metallica", "en-us")
        self.assertIntentMatch(result, 'OCP_fallback', 'ocp:play')

    def test_handle_player_state_update(self):
        initial_state = self.ocp.player_state

        # Test player state update to PLAYING
        self.ocp.bus.emit(Message("ovos.common_play.player.state",
                                  data={"state": PlayerState.PLAYING}))
        self.assertEqual(self.ocp.player_state, PlayerState.PLAYING)

        # Test player state update to PAUSED
        self.ocp.bus.emit(Message("ovos.common_play.player.state",
                                  data={"state": PlayerState.PAUSED}))
        self.assertEqual(self.ocp.player_state, PlayerState.PAUSED)

        # Test player state update to STOPPED

        self.ocp.bus.emit(Message("ovos.common_play.player.state",
                                  data={"state": PlayerState.STOPPED}))
        self.assertEqual(self.ocp.player_state, PlayerState.STOPPED)

        # Reset player state to the initial state for further testing
        self.ocp.player_state = initial_state

    def test_padacioso_intents(self):
        # Mock pipeline_engines
        lang = "en-us"

        # Test limited intents when player state is STOPPED
        self.ocp.player_state = PlayerState.STOPPED

        result = self.ocp.match_high("play metallica", lang)
        self.assertIsInstance(result, IntentMatch)
        self.assertEqual(result.intent_type, 'ocp:play')

        result = self.ocp.match_high("open OCP menu", lang)
        self.assertIsInstance(result, IntentMatch)
        self.assertEqual(result.intent_type, 'ocp:homescreen')

        # verify playback time intents dont match
        result = self.ocp.match_high("pause", lang)
        self.assertIsNone(result)
        result = self.ocp.match_high("next", lang)
        self.assertIsNone(result)
        result = self.ocp.match_high("previous", lang)
        self.assertIsNone(result)
        result = self.ocp.match_high("resume", lang)
        self.assertIsNone(result)  # TODO intent handler

        # Test when player in use new intents become available
        self.ocp.player_state = PlayerState.PAUSED

        result = self.ocp.match_high("pause", lang)
        self.assertIsInstance(result, IntentMatch)
        self.assertEqual(result.intent_type, 'ocp:pause')
        result = self.ocp.match_high("next tune", lang)
        self.assertIsInstance(result, IntentMatch)
        self.assertEqual(result.intent_type, 'ocp:next')
        result = self.ocp.match_high("previous song", lang)
        self.assertIsInstance(result, IntentMatch)
        self.assertEqual(result.intent_type, 'ocp:prev')
        result = self.ocp.match_high("play", lang)
        self.assertIsInstance(result, IntentMatch)
        self.assertEqual(result.intent_type, 'ocp:resume')
        result = self.ocp.match_high("resume", lang)
        self.assertIsInstance(result, IntentMatch)
        self.assertEqual(result.intent_type, 'ocp:resume')
        result = self.ocp.match_high("unpause", lang)
        self.assertIsInstance(result, IntentMatch)
        self.assertEqual(result.intent_type, 'ocp:resume')

        # Reset player state for further testing
        self.ocp.player_state = PlayerState.STOPPED


if __name__ == '__main__':
    unittest.main()
