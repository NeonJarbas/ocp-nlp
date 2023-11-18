from os.path import join, dirname, isfile

from ovos_utils.gui import can_use_gui
from ovos_utils.log import LOG
from ovos_workshop.decorators.ocp import *
from padacioso import IntentContainer


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

    def __init__(self, bus):
        self.bus = bus
        self.media_intents = IntentContainer()
        self.register_ocp_api_events()
        self.register_media_intents()

    def register_ocp_api_events(self):
        """
        Register messagebus handlers for OCP events
        """
        self.bus.on("ovos.common_play.search", self.handle_play)

    def register_ocp_intents(self, message=None):
        # TODO
        self.register_intent("play.intent", self.handle_play)
        self.register_intent("read.intent", self.handle_read)
        self.register_intent("open.intent", self.handle_open)
        self.register_intent("next.intent", self.handle_next)
        self.register_intent("prev.intent", self.handle_prev)
        self.register_intent("pause.intent", self.handle_pause)
        self.register_intent("resume.intent", self.handle_resume)

    def register_media_intents(self):
        """
        NOTE: uses the same format as mycroft .intent files, language
        support is handled the same way
        """
        locale_folder = join(dirname(__file__), "res", "locale", self.lang)
        intents = self.intent2media
        if self.settings.get("adult_content", False):
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

    # intents pipeline
    def handle_play(self, message):
        utterance = message.data["utterance"]
        phrase = message.data.get("query", "") or utterance
        LOG.debug(f"Handle {message.msg_type} request: {phrase}")
        num = message.data.get("number", "")
        if num:
            phrase += " " + num

        # if media is currently paused, empty string means "resume playback"
        if self._should_resume(phrase):
            self.player.resume()
            return
        if not phrase:
            phrase = self.get_response("play.what")
            if not phrase:
                # TODO some dialog ?
                self.player.stop()
                self.gui.show_home(app_mode=True)
                return

        # classify the query media type
        media_type = self.classify_media(utterance)

        # search common play skills
        results = self._search(phrase, utterance, media_type)
        self._do_play(phrase, results, media_type)

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

    # helper methods
    def _do_play(self, phrase, results, media_type=MediaType.GENERIC):
        self.player.reset()
        LOG.debug(f"Playing {len(results)} results for: {phrase}")
        if not results:
            if self.gui:
                if self.gui.active_extension == "smartspeaker":
                    self.gui.display_notification("Sorry, no matches found", style="warning")

            self.speak_dialog("cant.play",
                              data={"phrase": phrase,
                                    "media_type": media_type})

            if self.gui:
                if "smartspeaker" not in self.gui.active_extension:
                    if not self.gui.persist_home_display:
                        self.gui.remove_homescreen()
                    else:
                        self.gui.remove_search_spinner()
                else:
                    self.gui.clear_notification()

        else:
            if self.gui:
                if self.gui.active_extension == "smartspeaker":
                    self.gui.display_notification("Found a match", style="success")

            best = self.player.media.select_best(results)
            self.player.play_media(best, results)

            if self.gui:
                if self.gui.active_extension == "smartspeaker":
                    self.gui.clear_notification()

            self.enclosure.mouth_reset()  # TODO display music icon in mk1
            self.set_context("Playing")

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
        for r in self.player.media.search(phrase, media_type=media_type):
            results += r["results"]
        LOG.debug(f"Got {len(results)} results")
        # ignore very low score matches
        results = [r for r in results
                   if r["match_confidence"] >= self.settings.get("min_score",
                                                                 50)]
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

    def _should_resume(self, phrase: str) -> bool:
        """
        Check if a "play" request should resume playback or be handled as a new
        session.
        @param phrase: Extracted playback phrase
        @return: True if player should resume, False if this is a new request
        """
        if self.player.state == PlayerState.PAUSED:
            if not phrase.strip() or \
                    self.voc_match(phrase, "Resume", exact=True) or \
                    self.voc_match(phrase, "Play", exact=True):
                return True
        return False
