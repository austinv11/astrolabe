"""
Shared schema constants used across astrolabe submodules.

These cover identity strings that appear in multiple independent modules
and would otherwise require a multi-file grep to rename.  They are
intentionally kept to a small, stable set.
"""

SCHEMA_VERSION = "0.1"
UNS_SCHEMA_VERSION_KEY = "astrolabe_schema_version"
UNS_DEFAULT_MODALITY_KEY = "default_modality"
