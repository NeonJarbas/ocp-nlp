import mimetypes


def find_mime(uri):
    """ Determine mime type. """
    mime = mimetypes.guess_type(uri)
    if mime:
        return mime
    else:
        return None
