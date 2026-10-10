"""Dataset acquisition and preparation utilities.

Code lives in the package; the artifacts it downloads (shards, tokenizers,
``manifest.json``) are materialized under ``./data`` in the working directory
(override with ``VITRUM_DATA_DIR``), matching the defaults in
:class:`vitrum.core.config.DataConfig`.
"""
