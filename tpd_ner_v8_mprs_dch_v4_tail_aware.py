"""Compatibility wrapper for the integrated V4 tail-aware NER model.

The authoritative implementation lives in
``model.tpd_ner_v8_mprs_dch_v4_tail_aware``.  Re-exporting the same objects
keeps the original root-level reference import usable without creating a
second class identity.
"""

from model.tpd_ner_v8_mprs_dch_v4_tail_aware import *  # noqa: F401,F403
from model.tpd_ner_v8_mprs_dch_v4_tail_aware import __all__
