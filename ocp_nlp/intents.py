import os
import random
import re
from os.path import join, dirname
from threading import RLock

from ovos_bus_client.message import Message, dig_for_message
from ovos_core.intent_services import IntentMatch
from ovos_utils.enclosure.api import EnclosureAPI
from ovos_utils.log import LOG
from ovos_utils.messagebus import FakeBus
from padacioso import IntentContainer

from ocp_nlp.classify import BinaryPlaybackClassifier, BiasedMediaTypeClassifier, MediaTypeClassifier
from ocp_nlp.constants import OCP_ID, MediaType, PlaybackType, PlaybackMode, PlayerState
from ocp_nlp.search import OCPQuery


class OCPPipelineMatcher:

    def __init__(self, bus=None, config=None, entities_path=f"{dirname(__file__)}/sparql_ocp"):
        self.bus = bus or FakeBus()
        self.entities_path = entities_path
        # TODO - auto download OCP dataset to a XDG directory

        self.ocp_clfs = {}
        self.m_clfs = {}

        self._dialogs = {}  # lang: {name: [utts]}
        self._intents = {}  # lang: {name: [utts]}
        self._vocs = {}  # lang: {name: [utts]}

        self.config = config or {}
        self.search_lock = RLock()
        self.player_state = PlayerState.STOPPED  # TODO - track state via bus

        self.pipeline_engines = {}

        self.enclosure = EnclosureAPI(self.bus, skill_id=OCP_ID)

        self.load_resource_files()
        self.register_ocp_api_events()
        self.register_ocp_intents()

    def load_resource_files(self):
        # TODO - filter by native languages
        langs = ["en-us"]
        for lang in langs:
            self._intents[lang] = {}
            self._dialogs[lang] = {}
            self._vocs[lang] = {}
            locale_folder = join(dirname(__file__), "locale", lang)
            for f in os.listdir(locale_folder):
                path = join(locale_folder, f)
                if f.endswith(".dialog"):
                    with open(path) as intent:
                        samples = intent.read().split("\n")
                        for idx, s in enumerate(samples):
                            samples[idx] = s.replace("{{", "{").replace("}}", "}")
                        self._dialogs[lang][f] = samples
                if f.endswith(".intent"):
                    with open(path) as intent:
                        samples = intent.read().split("\n")
                        for idx, s in enumerate(samples):
                            samples[idx] = s.replace("{{", "{").replace("}}", "}")
                        self._intents[lang][f] = samples
                if f.endswith(".voc"):
                    with open(path) as intent:
                        samples = intent.read().split("\n")
                        self._vocs[lang][f] = samples

    def register_ocp_api_events(self):
        """
        Register messagebus handlers for OCP events
        """
        self.bus.on("ovos.common_play.search", self.handle_search_query)

    def register_ocp_intents(self):
        intents = ["play.intent", "open.intent",
                   "next.intent", "prev.intent", "pause.intent",
                   "resume.intent"]

        for lang, intent_data in self._intents.items():
            self.pipeline_engines[lang] = IntentContainer()
            for intent_name in intents:
                samples = intent_data.get(intent_name)
                LOG.debug(f"registering OCP intent: {intent_name}")
                self.pipeline_engines[lang].add_intent(
                    intent_name.replace(".intent", ""), samples)

    def load_clf(self, lang):
        if lang not in self.ocp_clfs:
            ocp_clf = BinaryPlaybackClassifier()
            ocp_clf.load()
            self.ocp_clfs[lang] = ocp_clf

        if lang not in self.m_clfs:
            clf1 = MediaTypeClassifier()
            clf1.load()
            clf = BiasedMediaTypeClassifier(clf1, lang="en", preload=True,
                                            dataset_path=self.entities_path)  # load entities database
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

        return IntentMatch(intent_service="OCP_intents",
                           intent_type=f'ocp:{match["name"]}',  # TODO intent event handler
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
        if self._should_resume(utterance, lang):
            # self.bus.emit(Message('ovos.common_play.resume'))
            return IntentMatch(intent_service="OCP_intents",
                               intent_type=f"ocp:resume",  # TODO intent event handler
                               intent_data=match,
                               skill_id=OCP_ID,
                               utterance=utterance)

        if not utterance:
            # user just said "play", we missed the search query somehow
            phrase = self.get_response("play.what")  # TODO - port this method
            if not phrase:
                # TODO some dialog ?
                # self.bus.emit(Message('ovos.common_play.stop'))
                return IntentMatch(intent_service="OCP_intents",
                                   intent_type=f"ocp:stop",  # TODO intent event handler
                                   intent_data=match,
                                   skill_id=OCP_ID,
                                   utterance=utterance)

        # classify the query media type
        media_type = self.classify_media(utterance, lang)

        # extract the query string
        query = self.remove_voc(utterance, "Play", lang).strip()

        ocp_clf, clf = self.load_clf(lang)
        ents = clf.extract_entities(utterance)

        # search common play skills
        # results = self._search(utterance, media_type, lang)
        # self._do_play(utterance, lang, results, media_type)
        return IntentMatch(intent_service="OCP_intents",
                           intent_type=f"ocp:play",  # TODO intent event handler
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

        self._process_play_query(phrase, lang)

    # intent handlers
    def _do_play(self, phrase, lang, results, media_type=MediaType.GENERIC):
        self.bus.emit(Message('ovos.common_play.reset'))
        LOG.debug(f"Playing {len(results)} results for: {phrase}")
        if not results:
            self.speak_dialog("cant.play", lang=lang,
                              data={"phrase": phrase,
                                    "media_type": media_type})
        else:
            best = self.select_best(results)
            results = [r for r in results if r != best]
            results.insert(0, best)
            self.bus.emit(Message("ovos.common_play.play",
                                  {"media": best, "disambiguation": results}))
            self.enclosure.mouth_reset()  # TODO display music icon in mk1
            self.bus.emit(Message('add_context',
                                  {'context': "Playing",
                                   'word': "",
                                   'origin': OCP_ID}))

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
        if self.player_state == PlayerState.PAUSED:  # TODO - track state via bus
            if not phrase.strip() or \
                    self.voc_match(phrase, "Resume", lang=lang, exact=True) or \
                    self.voc_match(phrase, "Play", lang=lang, exact=True):
                return True
        return False

    def voc_match(self, utt: str, voc_filename: str, lang: str, exact: bool = False):
        """
        Determine if the given utterance contains the vocabulary provided.

        By default the method checks if the utterance contains the given vocab
        thereby allowing the user to say things like "yes, please" and still
        match against "Yes.voc" containing only "yes". An exact match can be
        requested.

        The method first checks in the current Skill's .voc files and secondly
        in the "res/text" folder of mycroft-core. The result is cached to
        avoid hitting the disk each time the method is called.

        Args:
            utt (str): Utterance to be tested
            voc_filename (str): Name of vocabulary file (e.g. 'yes' for
                                'res/text/en-us/yes.voc')
            lang (str): Language code, defaults to self.lang
            exact (bool): Whether the vocab must exactly match the utterance

        Returns:
            bool: True if the utterance has the given vocabulary it
        """
        match = False
        if lang not in self._vocs:
            return False

        _vocs = self._vocs[lang].get(voc_filename) or \
                self._vocs[lang].get(voc_filename + ".voc") or \
                []

        if utt and _vocs:
            if exact:
                # Check for exact match
                match = any(i.strip() == utt
                            for i in _vocs)
            else:
                # Check for matches against complete words
                match = any([re.match(r'.*\b' + i + r'\b.*', utt)
                             for i in _vocs])

        return match

    def remove_voc(self, utt: str, voc_filename: str, lang: str) -> str:
        """
        Removes any vocab match from the utterance.
        @param utt: Utterance to evaluate
        @param voc_filename: vocab resource to remove from utt
        @param lang: Optional language associated with vocab and utterance
        @return: string with vocab removed
        """
        if lang not in self._vocs:
            return utt

        _vocs = self._vocs[lang].get(voc_filename) or \
                self._vocs[lang].get(voc_filename + ".voc") or \
                []
        if utt:
            # Check for matches against complete words
            for i in _vocs:
                # Substitute only whole words matching the token
                utt = re.sub(r'\b' + i + r"\b", "", utt)
        return utt

    def speak_dialog(self, dialog: str, data: dict, lang: str):
        samples = self._dialogs.get("lang", {}).get(dialog) or \
                  self._dialogs.get("lang", {}).get(dialog + ".dialog")
        if not samples:
            utt = dialog
        else:
            utt = random.choice(samples)
        data = {"utterance": utt, "lang": lang,
                "meta": {'dialog': dialog.replace(".dialog", ""),
                         'data': data}}
        # no dialog renderer, do a manual replace, accounting for whitespaces and double brackets
        for k, v in data.items():
            if isinstance(v, str):
                utt = utt.replace("{{", "{"). \
                    replace("}}", "}"). \
                    replace("{ ", "{"). \
                    replace(" }", "}"). \
                    replace("{" + k + "}", v)
        # grab message that triggered speech so we can keep context
        message = dig_for_message()
        m = message.forward("speak", data) if message \
            else Message("speak", data)
        m.context["skill_id"] = OCP_ID
        self.bus.emit(m)

    def get_response(self, dialog):
        return None  # TODO port this method from workshop

    # search
    def _search(self, phrase, media_type, lang: str):

        self.enclosure.mouth_think()
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

    ocp = OCPPipelineMatcher()
    print(ocp.match_high("play metallica", "en-us"))
    # IntentMatch(intent_service='OCP_intents',
    #   intent_type='ocp:play',
    #   intent_data={'media_type': <MediaType.MUSIC: 2>, 'query': 'metallica',
    #                'entities': {'album_name': 'Metallica', 'artist_name': 'Metallica'},
    #                'conf': 0.96, 'lang': 'en-us'},
    #   skill_id='ovos.common_play', utterance='play metallica')
    print(ocp.match_high("put on some metallica", "en-us"))
    # None

    print(ocp.match_medium("put on some metallica", "en-us"))
    # IntentMatch(intent_service='OCP_media',
    #   intent_type='ocp:play',
    #   intent_data={'media_type': <MediaType.MUSIC: 2>,
    #                'entities': {'album_name': 'Metallica', 'artist_name': 'Metallica', 'movie_name': 'Some'},
    #                'query': 'put on some metallica',
    #                'conf': 0.9578441098114333},
    #   skill_id='ovos.common_play', utterance='put on some metallica')
    print(ocp.match_medium("i wanna hear metallica", "en-us"))
    # None

    print(ocp.match_fallback("i wanna hear metallica", "en-us"))
    #  IntentMatch(intent_service='OCP_fallback',
    #    intent_type='ocp:play',
    #    intent_data={'media_type': <MediaType.MUSIC: 2>,
    #                 'entities': {'album_name': 'Metallica', 'artist_name': 'Metallica'},
    #                 'query': 'i wanna hear metallica',
    #                 'conf': 0.5027561091821287},
    #    skill_id='ovos.common_play', utterance='i wanna hear metallica')
