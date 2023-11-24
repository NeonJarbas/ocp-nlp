import mimetypes
from os.path import join, dirname
from typing import Optional, Tuple, List, Union

from ovos_utils.json_helper import merge_dict
from ovos_utils.log import LOG

from ocp_nlp.constants import *


def find_mime(uri):
    """ Determine mime type. """
    mime = mimetypes.guess_type(uri)
    if mime:
        return mime
    else:
        return None


# TODO subclass from dict (?)
class MediaEntry:
    def __init__(self, title="", uri="", skill_id=OCP_ID,
                 image=None, match_confidence=0,
                 playback=PlaybackType.UNDEFINED,
                 status=TrackState.DISAMBIGUATION, phrase=None,
                 position=0, length=None, bg_image=None, skill_icon=None,
                 artist=None, is_cps=False, cps_data=None, javascript="",
                 **kwargs):
        self.match_confidence = match_confidence
        self.title = title
        uri = uri or ""  # handle None
        self.uri = f'file://{uri}' if uri.startswith('/') else uri
        self.artist = artist
        self.skill_id = skill_id
        self.status = status
        self.playback = PlaybackType(playback) if isinstance(playback, int) \
            else playback
        self.image = image or join(dirname(__file__),
                                   "res/ui/images/ocp_bg.png")
        self.position = position
        self.phrase = phrase
        self.length = length  # None -> live stream
        self.skill_icon = skill_icon or join(dirname(__file__),
                                             "res/ui/images/ocp.png")
        self.bg_image = bg_image or join(dirname(__file__),
                                         "res/ui/images/ocp_bg.png")
        self.is_cps = is_cps
        self.data = kwargs
        self.cps_data = cps_data or {}
        self.javascript = javascript  # custom code to run in Webview after page load

    def update(self, entry: dict, skipkeys: list = None, newonly: bool = False):
        """
        Update this MediaEntry object with keys from the provided entry
        @param entry: dict or MediaEntry object to update this object with
        @param skipkeys: list of keys to not change
        @param newonly: if True, only adds new keys; existing keys are unchanged
        """
        skipkeys = skipkeys or []
        if isinstance(entry, MediaEntry):
            entry = entry.as_dict
        entry = entry or {}
        for k, v in entry.items():
            if k not in skipkeys and hasattr(self, k):
                if newonly and self.__getattribute__(k):
                    # skip, do not replace existing values
                    continue
                self.__setattr__(k, v)

    @staticmethod
    def from_dict(data: dict):
        """
        Construct a `MediaEntry` object from dict data.
        @param data: dict information to build the `MediaEntry` for
        @return: MediaEntry object
        """
        if data.get("bg_image") and data["bg_image"].startswith("/"):
            data["bg_image"] = "file:/" + data["bg_image"]
        data["skill"] = data.get("skill_id") or OCP_ID
        data["position"] = data.get("position", 0)
        data["length"] = data.get("length") or \
                         data.get("track_length") or \
                         data.get("duration")  # or get_duration_from_url(url)
        data["skill_icon"] = data.get("skill_icon") or data.get("skill_logo")
        data["status"] = data.get("status") or TrackState.DISAMBIGUATION
        data["playback"] = data.get("playback", PlaybackType.UNDEFINED)
        data["uri"] = data.get("stream") or data.get("uri") or data.get("url")
        data["title"] = data.get("title") or data["uri"]
        data["artist"] = data.get("artist") or data.get("author")
        data["is_cps"] = data.get("is_old_style") or data.get("is_cps", False)
        data["cps_data"] = data.get("cps_data") or {}
        return MediaEntry(**data)

    @property
    def info(self) -> dict:
        """
        Return a dict representation of this MediaEntry + infocard for QML model
        """
        return merge_dict(self.as_dict, self.infocard)

    @property
    def infocard(self) -> dict:
        """
        Return dict data used for a UI display
        """
        return {
            "duration": self.length,
            "track": self.title,
            "image": self.image,
            "album": self.skill_id,
            "source": self.skill_icon,
            "uri": self.uri
        }

    @property
    def mpris_metadata(self) -> dict:
        """
        Return dict data used by MPRIS
        """
        from dbus_next.service import Variant
        meta = {"xesam:url": Variant('s', self.uri)}
        if self.artist:
            meta['xesam:artist'] = Variant('as', [self.artist])
        if self.title:
            meta['xesam:title'] = Variant('s', self.title)
        if self.image:
            meta['mpris:artUrl'] = Variant('s', self.image)
        if self.length:
            meta['mpris:length'] = Variant('d', self.length)
        return meta

    @property
    def as_dict(self) -> dict:
        """
        Return a dict reporesentation of this MediaEntry
        """
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith("_")}

    @property
    def mimetype(self) -> Optional[Tuple[Optional[str], Optional[str]]]:
        """
        Get the detected mimetype tuple (type, encoding) if it can be determined
        """
        if self.uri:
            return find_mime(self.uri)

    def __eq__(self, other):
        if isinstance(other, MediaEntry):
            other = other.infocard
        # dict compatison
        return other == self.infocard

    def __repr__(self):
        return str(self.as_dict)

    def __str__(self):
        return str(self.as_dict)


class Playlist(list):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._position = 0

    @property
    def position(self) -> int:
        """
        Return the current position in the playlist
        """
        return self._position

    @property
    def entries(self) -> List[MediaEntry]:
        """
        Return a list of MediaEntry objects in the playlist
        """
        entries = []
        for e in self:
            if isinstance(e, dict):
                e = MediaEntry.from_dict(e)
            if isinstance(e, MediaEntry):
                entries.append(e)
        return entries

    @property
    def current_track(self) -> Optional[MediaEntry]:
        """
        Return the current MediaEntry or None if the playlist is empty
        """
        if len(self) == 0:
            return None
        self._validate_position()
        track = self[self.position]
        if isinstance(track, dict):
            track = MediaEntry.from_dict(track)
        return track

    @property
    def is_first_track(self) -> bool:
        """
        Return `True` if the current position is the first track or if the
        playlist is empty
        """
        if len(self) == 0:
            return True
        return self.position == 0

    @property
    def is_last_track(self) -> bool:
        """
        Return `True` if the current position is the last track of if the
        playlist is empty
        """
        if len(self) == 0:
            return True
        return self.position == len(self) - 1

    def goto_start(self) -> None:
        """
        Move to the first entry in the playlist
        """
        self._position = 0

    def clear(self) -> None:
        """
        Remove all entries from the Playlist and reset the position
        """
        super(Playlist, self).clear()
        self._position = 0

    def sort_by_conf(self):
        """
        Sort the Playlist by `match_confidence` with high confidence first
        """
        self.sort(key=lambda k: k.match_confidence
        if isinstance(k, MediaEntry) else
        k.get("match_confidence", 0), reverse=True)

    def add_entry(self, entry: MediaEntry, index: int = -1) -> None:
        """
        Add an entry at the requested index
        @param entry: MediaEntry to add to playlist
        @param index: index to insert entry at (default -1 to append)
        """
        assert isinstance(index, int)
        # TODO: Handle index out of range
        if isinstance(entry, dict):
            entry = MediaEntry.from_dict(entry)
        assert isinstance(entry, MediaEntry)
        if index == -1:
            index = len(self)

        if index < self.position:
            self.set_position(self.position + 1)

        self.insert(index, entry)

    def remove_entry(self, entry: Union[int, dict, MediaEntry]) -> None:
        """
        Remove the requested entry from the playlist or raise a ValueError
        @param entry: index or MediaEntry to remove from the playlist
        """
        if isinstance(entry, int):
            self.pop(entry)
            return
        if isinstance(entry, dict):
            entry = MediaEntry.from_dict(entry)
        assert isinstance(entry, MediaEntry)
        for idx, e in self.entries:
            if e == entry:
                self.pop(idx)
                break
        else:
            raise ValueError(f"entry not in playlist: {entry}")

    def replace(self, new_list: List[Union[dict, MediaEntry]]) -> None:
        """
        Replace the contents of this Playlist with new_list
        @param new_list: list of MediaEntry or dict objects to set this list to
        """
        self.clear()
        for e in new_list:
            self.add_entry(e)

    def set_position(self, idx: int):
        """
        Set the position in the playlist to a specific index
        @param idx: Index to set position to
        """
        self._position = idx
        self._validate_position()

    def goto_track(self, track: Union[MediaEntry, dict]) -> None:
        """
        Go to the requested track in the playlist
        @param track: MediaEntry to find and go to in the playlist
        """
        if isinstance(track, MediaEntry):
            requested_uri = track.uri
        else:
            requested_uri = track.get("uri", "")
        for idx, t in enumerate(self):
            if isinstance(t, MediaEntry):
                pl_entry_uri = t.uri
            else:
                pl_entry_uri = t.get("uri", "")
            if requested_uri == pl_entry_uri:
                self.set_position(idx)
                LOG.debug(f"New playlist position: {self.position}")
                return
        LOG.error(f"requested track not in the playlist: {track}")

    def next_track(self) -> None:
        """
        Go to the next track in the playlist
        """
        self.set_position(self.position + 1)

    def prev_track(self) -> None:
        """
        Go to the previous track in the playlist
        """
        self.set_position(self.position - 1)

    def _validate_position(self) -> None:
        """
        Make sure the current position is valid; default `position` to 0
        """
        if self.position < 0 or self.position >= len(self):
            LOG.error(f"Playlist pointer is in an invalid position "
                      f"({self.position}! Going to start of playlist")
            self._position = 0

    def __contains__(self, item):
        if isinstance(item, dict):
            item = MediaEntry.from_dict(item)
        if not isinstance(item, MediaEntry):
            return False
        for e in self.entries:
            if not e.uri and e.data.get("playlist"):
                if e.title == item.title and not item.uri:
                    return True
                # track in playlist
                for t in e.data["playlist"]:
                    if t.get("uri") == item.uri:
                        return True
            elif e.uri == item.uri:
                return True
        return False
