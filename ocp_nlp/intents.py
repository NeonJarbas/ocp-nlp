from os.path import join, dirname

import os
import random
from ovos_bus_client.message import Message
from ovos_utils.log import LOG
from ovos_utils.messagebus import FakeBus
from ovos_utils.skills.audioservice import OCPInterface
from padacioso import IntentContainer
from threading import RLock

from ocp_nlp.classify import BinaryPlaybackClassifier, BiasedMediaTypeClassifier, MediaTypeClassifier
from ocp_nlp.constants import OCP_ID, MediaType, PlaybackType, PlaybackMode, PlayerState
from ocp_nlp.search import OCPQuery
from ovos_core.intent_services import IntentMatch
from ovos_workshop.app import OVOSAbstractApplication


class OCPPipelineMatcher(OVOSAbstractApplication):

    def __init__(self, bus=None, config=None):
        super().__init__(skill_id=OCP_ID, bus=bus or FakeBus(),
                         resources_dir=f"{dirname(__file__)}")
        self.ocp_clfs = {}
        self.m_clfs = {}
        self.ocp_api = OCPInterface(self.bus)

        self.config = config or {}
        self.search_lock = RLock()
        self.player_state = PlayerState.STOPPED

        self.pipeline_engines = {}

        self.register_ocp_api_events()
        self.register_ocp_intents()

    def load_resource_files(self):
        intents = {}
        for lang in self.native_langs:
            intents[lang] = {}
            locale_folder = join(dirname(__file__), "locale", lang)
            for f in os.listdir(locale_folder):
                path = join(locale_folder, f)
                if f.endswith(".intent"):
                    with open(path) as intent:
                        samples = intent.read().split("\n")
                        for idx, s in enumerate(samples):
                            samples[idx] = s.replace("{{", "{").replace("}}", "}")
                        intents[lang][f] = samples
        return intents

    def register_ocp_api_events(self):
        """
        Register messagebus handlers for OCP events
        """
        self.bus.on("ovos.common_play.search", self.handle_search_query)
        self.bus.on('ovos.common_play.player.state', self.handle_player_state_update)

    def register_ocp_intents(self):
        intent_files = self.load_resource_files()

        intents = ["play.intent", "open.intent",
                   "next.intent", "prev.intent", "pause.intent",
                   "resume.intent", "stop.intent"]

        for lang, intent_data in intent_files.items():
            self.pipeline_engines[lang] = IntentContainer()
            for intent_name in intents:
                samples = intent_data.get(intent_name)
                LOG.debug(f"registering OCP intent: {intent_name}")
                self.pipeline_engines[lang].add_intent(
                    intent_name.replace(".intent", ""), samples)

        self.bus.on("ocp:play", self.handle_play_intent)
        self.bus.on("ocp:open", self.handle_open_intent)
        self.bus.on("ocp:next", self.handle_next_intent)
        self.bus.on("ocp:prev", self.handle_prev_intent)
        self.bus.on("ocp:pause", self.handle_pause_intent)
        self.bus.on("ocp:resume", self.handle_resume_intent)
        self.bus.on("ocp:stop", self.handle_stop_intent)
        self.bus.on("ocp:search_error", self.handle_search_error_intent)

    def handle_player_state_update(self, message):
        """
        Handles 'ovos.common_play.player.state' messages with player state updates
        @param message: Message providing new "state" data
        """
        state = message.data.get("state")
        if state is None:
            raise ValueError(f"Got state update message with no state: "
                             f"{message}")
        if isinstance(state, int):
            state = PlayerState(state)
        if not isinstance(state, PlayerState):
            raise ValueError(f"Expected int or PlayerState, but got: {state}")
        if state == self.player_state:
            return
        LOG.debug(f"PlayerState changed: {repr(state)}")
        if state == PlayerState.PLAYING:
            self.player_state = PlayerState.PLAYING
        elif state == PlayerState.PAUSED:
            self.player_state = PlayerState.PAUSED
        elif state == PlayerState.STOPPED:
            self.player_state = PlayerState.STOPPED

    def load_clf(self, lang):
        if lang not in self.ocp_clfs:
            ocp_clf = BinaryPlaybackClassifier()
            ocp_clf.load()
            self.ocp_clfs[lang] = ocp_clf

        if lang not in self.m_clfs:
            clf1 = MediaTypeClassifier()
            clf1.load()
            clf = BiasedMediaTypeClassifier(clf1, lang="en", preload=True)  # load entities database
            clf.load()
            self.m_clfs[lang] = clf
        return self.ocp_clfs[lang], self.m_clfs[lang]

    # pipeline
    def match_high(self, utterance, lang):
        """ exact matches only, handles playback control
        recommended after high confidence intents pipeline stage """
        if lang not in self.pipeline_engines:
            return None

        match = self.pipeline_engines[lang].calc_intent(utterance)

        if match["name"] is None:
            return None
        if match["name"] == "play":
            return self._process_play_query(utterance, lang, match)

        if self.player_state == PlayerState.STOPPED:
            # next / previous / pause / resume not targeted
            # at OCP if playback is not happening / paused
            if match["name"] == "resume":
                # TODO - handle resume for last_played query, eg, previous day
                return None
            elif match["name"] == "open":  # TODO check for gui connected
                pass  # open the GUI
            else:
                return None

        return IntentMatch(intent_service="OCP_intents",
                           intent_type=f'ocp:{match["name"]}',
                           intent_data=match,
                           skill_id=OCP_ID,
                           utterance=utterance)

    def match_medium(self, utterance, lang):
        """ match a utterance via classifiers,
        recommended before common_qa pipeline stage"""
        ocp_clf, clf = self.load_clf(lang)
        is_ocp = ocp_clf.predict([utterance])[0] == "OCP"
        if not is_ocp:
            return None
        label, confidence = clf.predict_prob([utterance])[0]
        mt = clf.label2media(label)
        ents = clf.extract_entities(utterance)
        # extract the query string
        query = self.remove_voc(utterance, "Play", lang).strip()
        return IntentMatch(intent_service="OCP_media",
                           intent_type=f"ocp:play",
                           intent_data={"media_type": mt,
                                        "entities": ents,
                                        "query": query,
                                        "conf": confidence},
                           skill_id=OCP_ID,
                           utterance=utterance)

    def match_fallback(self, utterance, lang):
        """ match a utterance via presence of known OCP keywords,
        recommended before fallback_low pipeline stage"""
        ocp_clf, clf = self.load_clf(lang)
        ents = clf.extract_entities(utterance)
        if not ents:
            return None
        label, confidence = clf.predict_prob([utterance])[0]
        mt = clf.label2media(label)
        # extract the query string
        query = self.remove_voc(utterance, "Play", lang).strip()
        return IntentMatch(intent_service="OCP_fallback",
                           intent_type=f"ocp:play",
                           intent_data={"media_type": mt,
                                        "entities": ents,
                                        "query": query,
                                        "conf": confidence},
                           skill_id=OCP_ID,
                           utterance=utterance)

    def _process_play_query(self, utterance, lang, match):
        # if media is currently paused, empty string means "resume playback"
        if self.player_state == PlayerState.PAUSED and \
                self._should_resume(utterance, lang):
            return IntentMatch(intent_service="OCP_intents",
                               intent_type=f"ocp:resume",
                               intent_data=match,
                               skill_id=OCP_ID,
                               utterance=utterance)

        if not utterance:
            # user just said "play", we are missing the search query
            phrase = self.get_response("play.what")
            if not phrase:
                # let the error intent handler take action
                return IntentMatch(intent_service="OCP_intents",
                                   intent_type=f"ocp:search_error",
                                   intent_data=match,
                                   skill_id=OCP_ID,
                                   utterance=utterance)

        # classify the query media type
        media_type = self.classify_media(utterance, lang)

        # extract the query string
        query = self.remove_voc(utterance, "Play", lang).strip()

        ocp_clf, clf = self.load_clf(lang)
        ents = clf.extract_entities(utterance)

        return IntentMatch(intent_service="OCP_intents",
                           intent_type=f"ocp:play",
                           intent_data={"media_type": media_type,
                                        "query": query,
                                        "entities": ents,
                                        "conf": match["conf"],
                                        # "results": results,
                                        "lang": lang},
                           skill_id=OCP_ID,
                           utterance=utterance)

    # bus api
    def handle_search_query(self, message):
        utterance = message.data["utterance"]
        phrase = message.data.get("query", "") or utterance
        lang = message.data.get("lang") or message.context.get("session", {}).get("lang", "en-us")
        LOG.debug(f"Handle {message.msg_type} request: {phrase}")
        num = message.data.get("number", "")
        if num:
            phrase += " " + num

        # classify the query media type
        media_type = self.classify_media(utterance, lang)

        # search common play skills
        results = self._search(phrase, media_type, lang)
        best = self.select_best(results)
        self.bus.emit(message.response(data={"results": results, "best": best}))

    # intent handlers
    def handle_play_intent(self, message: Message):
        lang = message.data["lang"]
        query = message.data["query"]
        media_type = message.data["media_type"]

        # search common play skills
        results = self._search(query, media_type, lang)

        # tell OCP to play
        self.bus.emit(Message('ovos.common_play.reset'))
        if not results:
            self.speak_dialog("cant.play",
                              data={"phrase": query,
                                    "media_type": media_type})
        else:
            LOG.debug(f"Playing {len(results)} results for: {query}")
            best = self.select_best(results)
            results = [r for r in results if r != best]
            results.insert(0, best)
            self.bus.emit(Message('add_context',
                                  {'context': "Playing",
                                   'word': "",
                                   'origin': OCP_ID}))

            # ovos-PHAL-plugin-mk1 will display music icon in response to play message
            self.ocp_api.play(results, query)

    def handle_open_intent(self, message: Message):
        pass  # TODO - show gui

    def handle_stop_intent(self, message: Message):
        self.ocp_api.stop()

    def handle_next_intent(self, message: Message):
        self.ocp_api.next()

    def handle_prev_intent(self, message: Message):
        self.ocp_api.prev()

    def handle_pause_intent(self, message: Message):
        self.ocp_api.pause()

    def handle_resume_intent(self, message: Message):
        self.ocp_api.resume()

    def handle_search_error_intent(self, message: Message):
        self.speak_dialog("play.not.understood")
        self.ocp_api.stop()

    def _do_play(self, phrase, results, media_type=MediaType.GENERIC):
        self.bus.emit(Message('ovos.common_play.reset'))
        LOG.debug(f"Playing {len(results)} results for: {phrase}")
        if not results:
            self.speak_dialog("cant.play",
                              data={"phrase": phrase,
                                    "media_type": media_type})
        else:
            best = self.select_best(results)
            results = [r for r in results if r != best]
            results.insert(0, best)
            self.bus.emit(Message('add_context',
                                  {'context': "Playing",
                                   'word': "",
                                   'origin': OCP_ID}))

            # ovos-PHAL-plugin-mk1 will display music icon in response to play message
            self.ocp_api.play(results, phrase)

    # NLP
    def classify_media(self, query, lang):
        """ determine what media type is being requested """

        if self.voc_match(query, "audio_only", lang=lang):
            query = self.remove_voc(query, "audio_only", lang=lang).strip()
        elif self.voc_match(query, "video_only", lang=lang):
            query = self.remove_voc(query, "video_only", lang=lang)

        ocp_clf, clf = self.load_clf(lang)
        label, confidence = clf.predict_prob([query])[0]
        LOG.info(f"OVOSCommonPlay MediaType prediction: {label}")
        LOG.debug(f"     utterance: {query}")
        return clf.label2media(label)

    def _should_resume(self, phrase: str, lang: str) -> bool:
        """
        Check if a "play" request should resume playback or be handled as a new
        session.
        @param phrase: Extracted playback phrase
        @return: True if player should resume, False if this is a new request
        """
        if self.player_state == PlayerState.PAUSED:
            if not phrase.strip() or \
                    self.voc_match(phrase, "Resume", lang=lang, exact=True) or \
                    self.voc_match(phrase, "Play", lang=lang, exact=True):
                return True
        return False

    # search
    def _search(self, phrase, media_type, lang: str):

        self.enclosure.mouth_think()  # animate mk1 mouth during search

        # check if user said "play XXX audio only/no video"
        audio_only = False
        video_only = False
        if self.voc_match(phrase, "audio_only", lang=lang):
            audio_only = True
            # dont include "audio only" in search query
            phrase = self.remove_voc(phrase, "audio_only", lang=lang)

        elif self.voc_match(phrase, "video_only", lang=lang):
            video_only = True
            # dont include "video only" in search query
            phrase = self.remove_voc(phrase, "video_only", lang=lang)

        # Now we place a query on the messsagebus for anyone who wants to
        # attempt to service a 'play.request' message.
        results = []
        for r in self._execute_query(phrase, media_type=media_type):
            results += r["results"]
        LOG.debug(f"Got {len(results)} results")

        # ignore very low score matches
        results = [r for r in results
                   if r["match_confidence"] >= self.config.get("min_score", 50)]
        LOG.debug(f"Got {len(results)} usable results")

        # check if user said "play XXX audio only"
        if audio_only:
            LOG.info("audio only requested, forcing audio playback "
                     "unconditionally")
            for idx, r in enumerate(results):
                # force streams to be played audio only
                results[idx]["playback"] = PlaybackType.AUDIO

        # check if user said "play XXX video only"
        elif video_only:
            LOG.info("video only requested, filtering non-video results")
            for idx, r in enumerate(results):
                if results[idx]["media_type"] == MediaType.VIDEO:
                    # force streams to be played in video mode, even if
                    # audio playback requested
                    results[idx]["playback"] = PlaybackType.VIDEO

            # filter audio only streams
            results = [r for r in results
                       if r["playback"] == PlaybackType.VIDEO]

        LOG.debug(f"Returning {len(results)} results")
        return results

    def _execute_query(self, phrase, media_type=MediaType.GENERIC):
        """ actually send the search to OCP skills"""
        with self.search_lock:
            # stop any search still happening
            self.bus.emit(Message("ovos.common_play.search.stop"))

            query = OCPQuery(query=phrase, media_type=media_type,
                             config=self.config, bus=self.bus)
            query.send()
            query.wait()

            # fallback to generic search type
            if not query.results and \
                    self.config.get("search_fallback", True) and \
                    media_type != MediaType.GENERIC:
                LOG.debug("OVOSCommonPlay falling back to MediaType.GENERIC")
                query.media_type = MediaType.GENERIC
                query.reset()
                query.send()
                query.wait()

        LOG.debug(f'Returning {len(query.results)} search results')
        return query.results

    def search_skill(self, skill_id, phrase,
                     media_type=MediaType.GENERIC):
        res = [r for r in self._execute_query(phrase, media_type)
               if r["skill_id"] == skill_id]
        if not len(res):
            return None
        return res[0]

    def select_best(self, results):
        # Look at any replies that arrived before the timeout
        # Find response(s) with the highest confidence
        best = None
        ties = []

        for res in results:
            if not best or res['match_confidence'] > best['match_confidence']:
                best = res
                ties = [best]
            elif res['match_confidence'] == best['match_confidence']:
                ties.append(res)

        if ties:
            # select randomly
            selected = random.choice(ties)

            if self.config.get("playback_mode") == PlaybackMode.VIDEO_ONLY:
                # select only from VIDEO results if preference is set
                vid_results = [r for r in ties if r["playback"] ==
                               PlaybackType.VIDEO]
                if len(vid_results):
                    selected = random.choice(vid_results)
                else:
                    return None
            elif self.config.get("playback_mode") == PlaybackMode.AUDIO_ONLY:
                # select only from AUDIO results if preference is set
                audio_results = [r for r in ties if r["playback"] !=
                                 PlaybackType.VIDEO]
                if len(audio_results):
                    selected = random.choice(audio_results)
                else:
                    return None

            # TODO: Ask user to pick between ties or do it automagically
        else:
            selected = best
        LOG.debug(f"OVOSCommonPlay selected: {selected['skill_id']} - "
                  f"{selected['match_confidence']}")
        return selected


if __name__ == "__main__":
    LOG.set_level("DEBUG")
    bus = FakeBus()

    ocp = OCPPipelineMatcher(bus=bus)

    print(ocp.match_high("play metallica", "en-us"))
    # IntentMatch(intent_service='OCP_intents',
    #   intent_type='ocp:play',
    #   intent_data={'media_type': <MediaType.MUSIC: 2>, 'query': 'metallica',
    #                'entities': {'album_name': 'Metallica', 'artist_name': 'Metallica'},
    #                'conf': 0.96, 'lang': 'en-us'},
    #   skill_id='ovos.common_play', utterance='play metallica')

    print(ocp.match_medium("put on some metallica", "en-us"))
    # IntentMatch(intent_service='OCP_media',
    #   intent_type='ocp:play',
    #   intent_data={'media_type': <MediaType.MUSIC: 2>,
    #                'entities': {'album_name': 'Metallica', 'artist_name': 'Metallica', 'movie_name': 'Some'},
    #                'query': 'put on some metallica',
    #                'conf': 0.9578441098114333},
    #   skill_id='ovos.common_play', utterance='put on some metallica')

    print(ocp.match_fallback("i wanna hear metallica", "en-us"))
    #  IntentMatch(intent_service='OCP_fallback',
    #    intent_type='ocp:play',
    #    intent_data={'media_type': <MediaType.MUSIC: 2>,
    #                 'entities': {'album_name': 'Metallica', 'artist_name': 'Metallica'},
    #                 'query': 'i wanna hear metallica',
    #                 'conf': 0.5027561091821287},
    #    skill_id='ovos.common_play', utterance='i wanna hear metallica')

    from skill_ovos_somafm import SomaFMSkill
    from skill_ovos_youtube import SimpleYoutubeSkill

    s = SomaFMSkill(skill_id="somafm.ovos", bus=bus)
    # s = YoutubeMusicSkill(skill_id="ytmus.ovos", bus=bus)
    s = SimpleYoutubeSkill(skill_id="yt.ovos", bus=bus)


    def on_m(m):
        print(m)


    # bus.on("message", on_m)

    bus.emit(Message("ocp:play", {"lang": "en-us",
                                  "query": "rock",
                                  "media_type": MediaType.RADIO}))
    bus.emit(Message("ocp:play", {"lang": "en-us",
                                  "query": "turtles",
                                  "media_type": MediaType.VIDEO}))
