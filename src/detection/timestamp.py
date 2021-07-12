from datetime import datetime

STRFTIME_FORMAT = '%y%m%d_%H%M%S'


def timestamp():
    """make a timestamp, using the current time

    Returns
    -------
    timestamp : str
        the current time,
        returned in the format ``dactiloscopia.timestamp.STRFTIME_FORMAT``
    """
    return datetime.now().strftime(STRFTIME_FORMAT)
