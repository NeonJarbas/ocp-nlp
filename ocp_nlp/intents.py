import random
from os.path import join, dirname, isfile
from threading import RLock

from ovos_utils.gui import can_use_gui
from ovos_utils.log import LOG
from padacioso import IntentContainer

from .constants import *
from .search import OCPQuery


class OCP:
    intent2media = {
        "music": MediaType.MUSIC,
        "video": MediaType.VIDEO,
        "audiobook": MediaType.AUDIOBOOK,
        "radio": MediaType.RADIO,
        "radio_drama": MediaType.RADIO_THEATRE,
        "game": MediaType.GAME,
        "tv": MediaType.TV,
        "podcast": MediaType.PODCAST,
        "news": MediaType.NEWS,
        "movie": MediaType.MOVIE,
        "short_movie": MediaType.SHORT_FILM,
        "silent_movie": MediaType.SILENT_MOVIE,
        "bw_movie": MediaType.BLACK_WHITE_MOVIE,
        "documentaries": MediaType.DOCUMENTARY,
        "comic": MediaType.VISUAL_STORY,
        "movietrailer": MediaType.TRAILER,
        "behind_scenes": MediaType.BEHIND_THE_SCENES,

    }
    # filtered content
    adultintents = {
        "porn": MediaType.ADULT,
        "hentai": MediaType.HENTAI
    }

    def __init__(self, bus, config=None):
        self.bus = bus
        self.config = config or {}
        self.search_lock = RLock()
        self.player_state = PlayerState.STOPPED  # TODO - track state via bus
        self.pipeline_intents = IntentContainer()
        self.media_intents = IntentContainer()
        self.register_ocp_api_events()
        self.register_ocp_intents()
        self.register_media_intents()

    def register_ocp_api_events(self):
        """
        Register messagebus handlers for OCP events
        """
        self.bus.on("ovos.common_play.search", self.handle_search_query)

    def register_ocp_intents(self, message=None):
        # TODO - all native languages
        langs = ["en-us"]
        intents = ["play", "read", "open", "next", "prev", "pause.", "resume."]
        for lang in langs:
            locale_folder = join(dirname(__file__), "locale", lang)
            for intent_name in intents:
                path = join(locale_folder, intent_name + ".intent")
                if not isfile(path):
                    continue
                with open(path) as intent:
                    samples = intent.read().split("\n")
                    for idx, s in enumerate(samples):
                        samples[idx] = s.replace("{{", "{").replace("}}", "}")
                LOG.debug(f"registering OCP intent: {intent_name}")
                self.pipeline_intents.add_intent(intent_name, samples)

    def register_media_intents(self):
        """
        NOTE: uses the same format as mycroft .intent files, language
        support is handled the same way
        """
        # TODO - all native languages
        langs = ["en-us"]
        for lang in langs:
            locale_folder = join(dirname(__file__), "locale", lang)
            intents = self.intent2media
            if self.config.get("adult_content", False):
                intents.update(self.adultintents)

            for intent_name in intents:
                path = join(locale_folder, intent_name + ".intent")
                if not isfile(path):
                    continue
                with open(path) as intent:
                    samples = intent.read().split("\n")
                    for idx, s in enumerate(samples):
                        samples[idx] = s.replace("{{", "{").replace("}}", "}")
                LOG.debug(f"registering media type intent: {intent_name}")
                self.media_intents.add_intent(intent_name, samples)

    # pipeline
    def match_utterance(self, utterance):
        is_intents = False  # TODO - match ocp intents. if play parse query

        is_play = False
        if is_play:
            return self._process_play_query(utterance)
        else:  # TODO - other intents
            pass

    def _process_play_query(self, utterance):
        phrase = utterance  # TODO remove the "play" from the consumed intent
        # if media is currently paused, empty string means "resume playback"
        if self._should_resume(phrase):
            self.bus.emit(Message('ovos.common_play.resume'))
            return  # TODO ret IntentMatch
        if not phrase:
            phrase = self.get_response("play.what")
            if not phrase:
                # TODO some dialog ?
                self.bus.emit(Message('ovos.common_play.stop'))
                return  # TODO ret IntentMatch

        # classify the query media type
        media_type = self.classify_media(utterance)

        # search common play skills
        results = self._search(phrase, utterance, media_type)
        self._do_play(phrase, results, media_type)
        return  # TODO ret IntentMatch

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
            self.bus.emit(Message("ovos.common_play.play",
                                  {"media": best, "disambiguation": results}))
            self.enclosure.mouth_reset()  # TODO display music icon in mk1
            self.set_context("Playing")

    # api
    def handle_search_query(self, message):
        utterance = message.data["utterance"]
        phrase = message.data.get("query", "") or utterance
        LOG.debug(f"Handle {message.msg_type} request: {phrase}")
        num = message.data.get("number", "")
        if num:
            phrase += " " + num

        self._process_play_query(phrase)

    # intents

    # "read XXX" - non "play XXX" audio book intent
    def handle_read(self, message):
        utterance = message.data["utterance"]
        phrase = message.data.get("query", "") or utterance
        # search common play skills
        results = self._search(phrase, utterance, MediaType.AUDIOBOOK)
        self._do_play(phrase, results, MediaType.AUDIOBOOK)

    # NLP
    def classify_media(self, query):
        """ this method uses a strict regex based parser to determine what
        media type is being requested, this helps in the media process
        - only skills that support media type are considered
        - if no matches a generic media is performed
        - some skills only answer for specific media types, usually to avoid over matching
        - skills may use media type to calc confidence
        - skills may ignore media type

        NOTE: uses the same format as mycroft .intent files, language
        support is handled the same way
        """
        if self.voc_match(query, "audio_only"):
            query = self.remove_voc(query, "audio_only").strip()
        elif self.voc_match(query, "video_only"):
            query = self.remove_voc(query, "video_only")

        pred = self.media_intents.calc_intent(query)
        LOG.info(f"OVOSCommonPlay MediaType prediction: {pred}")
        LOG.debug(f"     utterance: {query}")
        intent = pred.get("name", "")
        if intent in self.intent2media:
            return self.intent2media[intent]
        LOG.debug("Generic OVOSCommonPlay query")
        return MediaType.GENERIC

    def _should_resume(self, phrase: str) -> bool:
        """
        Check if a "play" request should resume playback or be handled as a new
        session.
        @param phrase: Extracted playback phrase
        @return: True if player should resume, False if this is a new request
        """
        if self.player_state == PlayerState.PAUSED:  # TODO - track state via bus
            if not phrase.strip() or \
                    self.voc_match(phrase, "Resume", exact=True) or \
                    self.voc_match(phrase, "Play", exact=True):
                return True
        return False

    # search
    def _search(self, phrase, utterance, media_type):
        self.enclosure.mouth_think()
        # check if user said "play XXX audio only/no video"
        audio_only = False
        video_only = False
        if self.voc_match(phrase, "audio_only"):
            audio_only = True
            # dont include "audio only" in search query
            phrase = self.remove_voc(phrase, "audio_only")
            # dont include "audio only" in media type classification
            utterance = self.remove_voc(utterance, "audio_only").strip()
        elif self.voc_match(phrase, "video_only"):
            video_only = True
            # dont include "video only" in search query
            phrase = self.remove_voc(phrase, "video_only")

        # Now we place a query on the messsagebus for anyone who wants to
        # attempt to service a 'play.request' message.
        results = []
        phrase = phrase or utterance
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

        # filter video results if GUI not connected
        elif not can_use_gui(self.bus):
            LOG.info("unable to use GUI, filtering non-audio results")
            # filter video only streams
            results = [r for r in results
                       if r["playback"] in [PlaybackType.AUDIO, PlaybackType.SKILL]]

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
