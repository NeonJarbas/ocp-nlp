import unittest
from unittest.mock import Mock
from ocp_nlp.constants import MediaType
from ocp_nlp.media import MediaEntry, Playlist  # Replace 'your_module' with the actual module name


class TestMediaEntry(unittest.TestCase):

    def test_init(self):
        media_entry = MediaEntry(title="Test Title", uri="test_uri", artist="Test Artist")
        self.assertEqual(media_entry.title, "Test Title")
        self.assertEqual(media_entry.uri, "test_uri")
        self.assertEqual(media_entry.artist, "Test Artist")
        self.assertEqual(media_entry.skill_id, "ocp_id")  # Default value
        self.assertEqual(media_entry.playback, MediaType.UNDEFINED)  # Default value
        self.assertEqual(media_entry.status, "disambiguation")  # Default value
        self.assertIsNone(media_entry.length)
        self.assertFalse(media_entry.is_cps)
        self.assertEqual(media_entry.cps_data, {})
        self.assertEqual(media_entry.javascript, "")

    def test_from_dict_with_bg_image(self):
        data = {
            "title": "Test Title",
            "uri": "test_uri",
            "bg_image": "/path/to/bg_image.jpg"
        }
        media_entry = MediaEntry.from_dict(data)
        self.assertEqual(media_entry.bg_image, "file://path/to/bg_image.jpg")

    def test_infocard(self):
        media_entry = MediaEntry(title="Test Title", uri="test_uri", artist="Test Artist")
        infocard = media_entry.infocard
        self.assertEqual(infocard["duration"], None)
        self.assertEqual(infocard["track"], "Test Title")
        self.assertEqual(infocard["image"], media_entry.image)
        self.assertEqual(infocard["album"], media_entry.skill_id)
        self.assertEqual(infocard["source"], media_entry.skill_icon)
        self.assertEqual(infocard["uri"], media_entry.uri)

    def test_mpris_metadata(self):
        media_entry = MediaEntry(title="Test Title", uri="test_uri", artist="Test Artist", length=300)
        mpris_metadata = media_entry.mpris_metadata
        self.assertEqual(mpris_metadata['xesam:url'], ('s', 'test_uri'))
        self.assertEqual(mpris_metadata['xesam:artist'], ('as', ['Test Artist']))
        self.assertEqual(mpris_metadata['xesam:title'], ('s', 'Test Title'))
        self.assertEqual(mpris_metadata['mpris:artUrl'], ('s', media_entry.image))
        self.assertEqual(mpris_metadata['mpris:length'], ('d', 300.0))

    def test_as_dict(self):
        media_entry = MediaEntry(title="Test Title", uri="test_uri", artist="Test Artist")
        as_dict = media_entry.as_dict
        self.assertEqual(as_dict["title"], "Test Title")
        self.assertEqual(as_dict["uri"], "test_uri")
        self.assertEqual(as_dict["artist"], "Test Artist")
        self.assertEqual(as_dict["length"], None)
        self.assertEqual(as_dict["is_cps"], False)
        self.assertEqual(as_dict["cps_data"], {})
        self.assertEqual(as_dict["javascript"], "")

    def test_eq(self):
        media_entry1 = MediaEntry(title="Test Title", uri="test_uri", artist="Test Artist")
        media_entry2 = MediaEntry(title="Test Title", uri="test_uri", artist="Test Artist")
        self.assertEqual(media_entry1, media_entry2)

    def test_repr(self):
        media_entry = MediaEntry(title="Test Title", uri="test_uri", artist="Test Artist")
        self.assertEqual(repr(media_entry), "{'title': 'Test Title', 'uri': 'test_uri', 'artist': 'Test Artist'}")

    def test_str(self):
        media_entry = MediaEntry(title="Test Title", uri="test_uri", artist="Test Artist")
        self.assertEqual(str(media_entry), "{'title': 'Test Title', 'uri': 'test_uri', 'artist': 'Test Artist'}")

    def test_from_dict(self):
        data = {
            "title": "Test Title",
            "uri": "test_uri",
            "artist": "Test Artist",
            "skill_id": "test_skill_id",
            "image": "test_image",
            "match_confidence": 0.8,
            "playback": MediaType.MUSIC,
            "status": "test_status",
            "phrase": "test_phrase",
            "position": 1,
            "length": 300,
            "bg_image": "test_bg_image",
            "skill_icon": "test_skill_icon",
            "is_cps": True,
            "cps_data": {"key": "value"},
            "javascript": "test_javascript"
        }
        media_entry = MediaEntry.from_dict(data)

        self.assertEqual(media_entry.title, "Test Title")
        self.assertEqual(media_entry.uri, "test_uri")
        self.assertEqual(media_entry.artist, "Test Artist")
        self.assertEqual(media_entry.skill_id, "test_skill_id")
        self.assertEqual(media_entry.image, "test_image")
        self.assertEqual(media_entry.match_confidence, 0.8)
        self.assertEqual(media_entry.playback, MediaType.MUSIC)
        self.assertEqual(media_entry.status, "test_status")
        self.assertEqual(media_entry.phrase, "test_phrase")
        self.assertEqual(media_entry.position, 1)
        self.assertEqual(media_entry.length, 300)
        self.assertEqual(media_entry.bg_image, "test_bg_image")
        self.assertEqual(media_entry.skill_icon, "test_skill_icon")
        self.assertTrue(media_entry.is_cps)
        self.assertEqual(media_entry.cps_data, {"key": "value"})
        self.assertEqual(media_entry.javascript, "test_javascript")

    def test_update(self):
        media_entry = MediaEntry(title="Old Title", uri="old_uri", artist="Old Artist")
        new_data = {"title": "New Title", "uri": "new_uri", "artist": "New Artist"}
        media_entry.update(new_data)

        self.assertEqual(media_entry.title, "New Title")
        self.assertEqual(media_entry.uri, "new_uri")
        self.assertEqual(media_entry.artist, "New Artist")

    def test_mimetype(self):
        media_entry = MediaEntry(uri="test.mp3")
        self.assertEqual(media_entry.mimetype, ("audio/mpeg", None))

    # Add more tests for other methods as needed


class TestPlaylist(unittest.TestCase):

    def test_add_entry(self):
        playlist = Playlist()
        entry = MediaEntry(title="Test Title", uri="test_uri")
        playlist.add_entry(entry)

        self.assertEqual(len(playlist), 1)
        self.assertEqual(playlist.entries[0].title, "Test Title")

    def test_remove_entry(self):
        playlist = Playlist()
        entry = MediaEntry(title="Test Title", uri="test_uri")
        playlist.add_entry(entry)

        self.assertEqual(len(playlist), 1)

        playlist.remove_entry(entry)
        self.assertEqual(len(playlist), 0)

    def test_replace(self):
        playlist = Playlist()
        entry1 = MediaEntry(title="Test Title 1", uri="test_uri_1")
        entry2 = MediaEntry(title="Test Title 2", uri="test_uri_2")
        new_list = [entry1, entry2]

        playlist.replace(new_list)
        self.assertEqual(len(playlist), 2)
        self.assertEqual(playlist.entries[0].title, "Test Title 1")
        self.assertEqual(playlist.entries[1].title, "Test Title 2")

    def test_goto_start(self):
        playlist = Playlist([{}, {}, {}])
        playlist.set_position(2)
        playlist.goto_start()
        self.assertEqual(playlist.position, 0)

    def test_clear(self):
        playlist = Playlist([{}, {}, {}])
        playlist.clear()
        self.assertEqual(len(playlist), 0)
        self.assertEqual(playlist.position, 0)

    def test_sort_by_conf(self):
        entry1 = MediaEntry(match_confidence=0.8)
        entry2 = MediaEntry(match_confidence=0.5)
        entry3 = MediaEntry(match_confidence=0.9)
        playlist = Playlist([entry1, entry2, entry3])
        playlist.sort_by_conf()
        self.assertEqual(playlist.entries[0].match_confidence, 0.9)
        self.assertEqual(playlist.entries[1].match_confidence, 0.8)
        self.assertEqual(playlist.entries[2].match_confidence, 0.5)

    def test_remove_entry_by_index(self):
        playlist = Playlist([{}, {}, {}])
        playlist.remove_entry(1)
        self.assertEqual(len(playlist), 2)

    def test_remove_entry_by_media_entry(self):
        entry1 = MediaEntry(title="Test Title 1", uri="test_uri_1")
        entry2 = MediaEntry(title="Test Title 2", uri="test_uri_2")
        playlist = Playlist([entry1, entry2])
        playlist.remove_entry(entry1)
        self.assertEqual(len(playlist), 1)
        self.assertEqual(playlist.entries[0].title, "Test Title 2")


if __name__ == '__main__':
    unittest.main()
