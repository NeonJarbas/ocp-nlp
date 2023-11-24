import time
from threading import Lock
from typing import List

from ovos_utils.gui import is_gui_connected, is_gui_running
from ovos_utils.log import LOG
from ovos_utils.messagebus import Message

from ocp_nlp.constants import *


def available_extractors():
    from ovos_plugin_common_play.ocp.utils import available_extractors as _real
    return _real() # TODO


class OCPQuery:
    cast2audio = [
        MediaType.MUSIC,
        MediaType.PODCAST,
        MediaType.AUDIOBOOK,
        MediaType.RADIO,
        MediaType.RADIO_THEATRE,
        MediaType.VISUAL_STORY,
        MediaType.NEWS
    ]

    def __init__(self, query, bus, media_type=MediaType.GENERIC, config=None):
        LOG.debug(f"Created {media_type.name} query: {query}")
        self.query = query
        self.media_type = media_type
        self.bus = bus
        self.config = config or {}
        self.reset()

    def reset(self):
        self.active_skills = {}
        self.active_skills_lock = Lock()
        self.query_replies = []
        self.searching = False
        self.search_start = 0
        self.query_timeouts = self.config.get("min_timeout", 5)
        if self.config.get("playback_mode") in [PlaybackMode.AUDIO_ONLY]:
            self.has_gui = False
        else:
            self.has_gui = is_gui_running() or is_gui_connected(self.bus)

    def send(self):
        self.query_replies = []
        self.query_timeouts = self.config.get("min_timeout", 5)
        self.search_start = time.time()
        self.searching = True
        self.register_events()
        self.bus.emit(Message('ovos.common_play.query',
                              {"phrase": self.query,
                               "question_type": self.media_type}))

    def wait(self):
        # if there is no match type defined, lets increase timeout a bit
        # since all skills need to search
        if self.media_type == MediaType.GENERIC:
            timeout = self.config.get("max_timeout", 15) + 3  # timeout bonus
        else:
            timeout = self.config.get("max_timeout", 15)
        while self.searching and time.time() - self.search_start <= timeout:
            time.sleep(0.1)
        self.searching = False
        self.remove_events()

    @property
    def results(self) -> List[dict]:
        return [s for s in self.query_replies
                if s.get("results")]

    def register_events(self):
        LOG.debug("Registering Search Bus Events")
        self.bus.on("ovos.common_play.skill.search_start", self.handle_skill_search_start)
        self.bus.on("ovos.common_play.skill.search_end", self.handle_skill_search_end)
        self.bus.on("ovos.common_play.query.response", self.handle_skill_response)

    def remove_events(self):
        LOG.debug("Removing Search Bus Events")
        self.bus.remove_all_listeners("ovos.common_play.skill.search_start")
        self.bus.remove_all_listeners("ovos.common_play.skill.search_end")
        self.bus.remove_all_listeners("ovos.common_play.query.response")

    def handle_skill_search_start(self, message):
        skill_id = message.data["skill_id"]
        LOG.debug(f"{message.data['skill_id']} is searching")
        with self.active_skills_lock:
            if skill_id not in self.active_skills:
                self.active_skills[skill_id] = Lock()

    def handle_skill_response(self, message):
        search_phrase = message.data["phrase"]
        if search_phrase != self.query:
            # not an answer for this search query
            return
        timeout = message.data.get("timeout")
        skill_id = message.data['skill_id']
        # LOG.debug(f"OVOSCommonPlay result: {skill_id}")

        # in case this handler fires before the search start handler
        with self.active_skills_lock:
            if skill_id not in self.active_skills:
                self.active_skills[skill_id] = Lock()
        with self.active_skills[skill_id]:
            if message.data.get("searching"):
                # extend the timeout by N seconds
                if timeout and self.config.get("allow_extensions", True):
                    self.query_timeouts += timeout
                # else -> expired search

            else:
                # Collect replies until the timeout
                if not self.searching and not len(self.query_replies):
                    LOG.debug("  too late!! ignored in track selection process")
                    LOG.warning(
                        f"{message.data['skill_id']} is not answering fast "
                        "enough!")

                # populate search playlist
                results = message.data.get("results", [])
                for idx, res in enumerate(results):
                    if self.media_type not in [MediaType.ADULT, MediaType.HENTAI]:
                        # skip adult content results unless explicitly enabled
                        if not self.config.get("adult_content", False) and \
                                res.get("media_type", MediaType.GENERIC) in \
                                [MediaType.ADULT, MediaType.HENTAI]:
                            continue

                    # filter uris we can play, usually files and http streams, but some
                    # skills might return results that depend on additional packages,
                    # eg. soundcloud, rss, youtube, deezer....
                    uri = res.get("uri", "")
                    if res.get("playlist") and not uri:
                        res["playlist"] = [
                            r for r in res["playlist"]
                            if r.get("uri") and any(r.get("uri").startswith(e)
                                                    for e in
                                                    available_extractors())]
                        if not len(res["playlist"]):
                            results[idx] = None  # can't play this search result!
                            LOG.error(f"Empty playlist for {res}")
                            continue
                    elif uri and res.get("playback") not in [
                        PlaybackType.SKILL, PlaybackType.UNDEFINED] and \
                            not any(
                                uri.startswith(e) for e in available_extractors()):
                        results[idx] = None  # can't play this search result!
                        LOG.error(f"stream handler not available for {res}")
                        continue

                    # filter video results if GUI not connected
                    if not self.has_gui:
                        # force allowed stream types to be played audio only
                        if res.get("media_type", "") in self.cast2audio:
                            LOG.debug("unable to use GUI, "
                                      "forcing result to play audio only")
                            res["playback"] = PlaybackType.AUDIO
                            res["match_confidence"] -= 10
                            results[idx] = res

                # remove filtered results
                message.data["results"] = [r for r in results if r is not None]
                LOG.debug(f'got {len(message.data["results"])} results from {skill_id}')
                self.query_replies.append(message.data)

                # abort searching if we gathered enough results
                # TODO ensure we have a decent confidence match, if all matches
                #  are < 50% conf extend timeout instead
                if time.time() - self.search_start > self.query_timeouts:
                    if self.searching:
                        self.searching = False
                        LOG.debug("common play query timeout, parsing results")

                elif self.searching:
                    for res in message.data.get("results", []):
                        if res.get("match_confidence", 0) >= \
                                self.config.get("early_stop_thresh", 85):
                            # got a really good match, dont search further
                            LOG.info(
                                "Receiving very high confidence match, stopping "
                                "search early")

                            # allow other skills to "just miss"
                            early_stop_grace = \
                                self.config.get("early_stop_grace_period", 0.5)
                            if early_stop_grace:
                                LOG.debug(
                                    f"  - grace period: {early_stop_grace} seconds")
                                time.sleep(early_stop_grace)
                            self.searching = False
                            return

    def handle_skill_search_end(self, message):
        skill_id = message.data["skill_id"]
        LOG.debug(f"{message.data['skill_id']} finished search")
        with self.active_skills_lock:
            if skill_id in self.active_skills:
                with self.active_skills[skill_id]:
                    del self.active_skills[skill_id]

        # if this was the last skill end searching period
        time.sleep(0.5)
        # TODO this sleep is hacky, but avoids a race condition in
        # case some skill just decides to respond before the others even
        # acknowledge search is starting, this gives more than enough time
        # for self.active_skills to be populated, a better approach should
        # be employed but this works fine for now
        if not self.active_skills and self.searching:
            LOG.info("Received search responses from all skills!")
            self.searching = False
