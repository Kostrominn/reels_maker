"""Interview reels: a declarative pipeline that turns a filmed conversation into
vertical reels with exact subtitles.

A reel is described in ``plans.json`` in the time of the reference audio track
(the transcript time): which camera is on screen when, where the cuts are and
what each subtitle says. Everything else — the render, the checks of the
finished files and the publication pack — is done by this package.

Entry point: ``python -m interview_reels --help``.
"""

__version__ = "1.0.0"
