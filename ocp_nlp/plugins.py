from typing import List

_plugins = None


def _ocp_plugins():
    global _plugins
    from ovos_plugin_manager.ocp import StreamHandler
    _plugins = _plugins or StreamHandler()
    return _plugins


def available_extractors() -> List[str]:
    """
    Get a list of supported Stream Extractor Identifiers. Note that these look
    like but are not URI schemes.
    @return: List of supported SEI prefixes
    """
    return ["/", "http:", "https:", "file:"] + \
        [f"{sei}//" for sei in _ocp_plugins().supported_seis]
