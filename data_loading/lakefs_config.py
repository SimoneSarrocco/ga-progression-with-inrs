"""Resolve the optional lakeFS connection. Single source of truth for credentials.

lakeFS (https://lakefs.io) is an open-source versioning layer over an S3-compatible
bucket. It is OPTIONAL. With no credentials the code reads images straight from local
disk, which is the default and is all you need to reproduce the paper from your own
copy of the data. See docs/DATA.md.

Both the training loader (data_loading/dataset.py) and the pre-fetch script
(download_lakefs_data.py) call make_loader() here, so they always agree on which
credentials, which bucket and which cache directory are in use.

WHERE SETTINGS COME FROM

Two places, and environment variables win so you can keep secrets out of files:

  1. configs/lakefs_cfg.yaml     copy configs/lakefs_cfg.example.yaml and fill it in.
                                 Git-ignored. Override the path with the environment
                                 variable GAPINR_LAKEFS_CFG, or with `config_path`
                                 inside the `lakefs:` block of config_data.yaml.
  2. environment variables       GAPINR_LAKEFS_ENDPOINT, GAPINR_LAKEFS_ACCESS_KEY,
                                 GAPINR_LAKEFS_SECRET_KEY, GAPINR_LAKEFS_CA_BUNDLE,
                                 GAPINR_LAKEFS_REPO, GAPINR_LAKEFS_BRANCH,
                                 GAPINR_LAKEFS_CACHE.

Which repository, branch and cache directory to use comes from the `lakefs:` block of
the dataset section in configs/config_data.yaml, and the three GAPINR_LAKEFS_REPO /
_BRANCH / _CACHE variables override those.

HOW FAILURES ARE REPORTED

Nothing is silent. Either lakeFS is off (a printed note, then local disk), or it is on
and working, or you get an error naming the missing or unedited setting. Credentials
half-filled in used to fall through to boto3's default AWS credential chain and fail
much later with an opaque S3 error.

To check your settings before a long run:

    python download_lakefs_data.py --check
"""

import os
import yaml

from .lakefsloader import LakeFSLoader

# Canonical key names, with the older spellings this repository used still accepted.
_ALIASES = {
    'endpoint':   ('endpoint', 's3_endpoint', 'lakefs_endpoint'),
    'access_key': ('access_key', 'lakefs_access_key'),
    'secret_key': ('secret_key', 'lakefs_secret_key'),
    'ca_bundle':  ('ca_bundle', 'ca_path', 'lakefs_ca_path'),
}
_ENV = {
    'endpoint':   'GAPINR_LAKEFS_ENDPOINT',
    'access_key': 'GAPINR_LAKEFS_ACCESS_KEY',
    'secret_key': 'GAPINR_LAKEFS_SECRET_KEY',
    'ca_bundle':  'GAPINR_LAKEFS_CA_BUNDLE',
}
# Values shipped in the template. Left unedited they must not reach boto3.
_PLACEHOLDERS = {
    'https://your-lakefs-host:8000', 'your-lakefs-host', 'YOUR_ACCESS_KEY_ID',
    'YOUR_SECRET_ACCESS_KEY', 'your-lakefs-repo', 'changeme', '<fill me in>',
}

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CFG_PATH = os.path.join(REPO_ROOT, 'configs', 'lakefs_cfg.yaml')


class LakeFSConfigError(RuntimeError):
    """Raised when lakeFS is partly configured, so the intent was clearly to use it."""


def config_path(lakefs_section=None):
    """Path of the credentials file: GAPINR_LAKEFS_CFG, then `config_path` in the
    dataset's `lakefs:` block, then configs/lakefs_cfg.yaml."""
    env = os.environ.get('GAPINR_LAKEFS_CFG')
    if env:
        return env
    if lakefs_section and lakefs_section.get('config_path'):
        p = lakefs_section['config_path']
        return p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)
    return DEFAULT_CFG_PATH


def resolve_cache_path(lakefs_section):
    """Root of the local image cache. Files sit at <cache_path>/<branch>/data/...

    Used by the pre-fetch, by the loader, and by the local-disk path in dataset.py, so
    downloaded images land exactly where training looks for them whether or not lakeFS
    is switched on. The layout matches docs/DATA.md, so a cache filled by lakeFS keeps
    working when you later run with no credentials."""
    cache = (os.environ.get('GAPINR_LAKEFS_CACHE')
             or (lakefs_section or {}).get('cache_path')
             or './cache/faf_ga')
    return cache


def read_credentials(lakefs_section=None):
    """Return (credentials dict, path read, how). Values are '' when unset. Environment
    variables take precedence over the file."""
    path = config_path(lakefs_section)
    raw = {}
    if os.path.exists(path):
        with open(path, 'r') as f:
            raw = yaml.safe_load(f) or {}
    creds = {}
    for key, names in _ALIASES.items():
        value = ''
        for name in names:
            if raw.get(name):
                value = str(raw[name]).strip()
                break
        env = os.environ.get(_ENV[key])
        if env:
            value = env.strip()
        creds[key] = value
    return creds, path


def _check_usable(creds, path):
    """Return True when lakeFS should be used, False when it is simply switched off.
    Raise when the settings are half-done or still hold template values."""
    required = ('endpoint', 'access_key', 'secret_key')
    present = [k for k in required if creds[k]]
    if not present:
        return False

    placeholders = [k for k in required if creds[k] in _PLACEHOLDERS]
    if placeholders:
        raise LakeFSConfigError(
            f"lakeFS settings still contain the template values for "
            f"{', '.join(placeholders)}.\n"
            f"  Edit {path} (copy it from configs/lakefs_cfg.example.yaml if it does not\n"
            f"  exist yet), or set the matching GAPINR_LAKEFS_* environment variables.\n"
            f"  To use local disk instead, leave every value empty or delete the file.")

    missing = [k for k in required if not creds[k]]
    if missing:
        raise LakeFSConfigError(
            f"lakeFS is partly configured: {', '.join(missing)} is missing.\n"
            f"  endpoint, access_key and secret_key are all required.\n"
            f"  Fill them in at {path}, or set GAPINR_LAKEFS_ENDPOINT / "
            f"GAPINR_LAKEFS_ACCESS_KEY / GAPINR_LAKEFS_SECRET_KEY.\n"
            f"  To use local disk instead, leave all three empty.")
    return True


def make_loader(dataset_cfg, verbose=True):
    """Build a LakeFSLoader for a dataset section of config_data.yaml, or return None
    when lakeFS is not configured and images should be read from local disk.

    Raises LakeFSConfigError when the settings are half-done, rather than failing later
    inside boto3."""
    section = (dataset_cfg or {}).get('lakefs')
    if not section:
        return None

    creds, path = read_credentials(section)
    if not _check_usable(creds, path):
        if verbose:
            where = path if os.path.exists(path) else f"{path} (not present)"
            print(f"[lakeFS] not configured, reading images from local disk. "
                  f"To enable it, fill in {where}.")
        return None

    repo = os.environ.get('GAPINR_LAKEFS_REPO') or section.get('repo')
    branch = os.environ.get('GAPINR_LAKEFS_BRANCH') or section.get('branch')
    if not repo or not branch:
        raise LakeFSConfigError(
            "Credentials were found, but the `lakefs:` block in configs/config_data.yaml "
            "is missing `repo` or `branch`. Both are required to locate your data.")
    cache = resolve_cache_path(section)

    # An empty ca_bundle turns off TLS verification, so say so rather than doing it quietly.
    if not creds['ca_bundle'] and creds['endpoint'].startswith('https') and verbose:
        print("[lakeFS] no ca_bundle set, TLS certificate verification is OFF. "
              "Set ca_bundle to your CA file to enable it.")

    loader = LakeFSLoader(
        repo_name=repo,
        branch_id=branch,
        local_cache_path=cache,
        endpoint=creds['endpoint'],
        ca_path=creds['ca_bundle'] or None,
        access_key=creds['access_key'],
        secret_key=creds['secret_key'],
    )
    if verbose:
        print(f"[lakeFS] repo={repo} branch={branch} endpoint={creds['endpoint']} "
              f"cache={cache}  (credentials from {path})")
    return loader


def check_connection(dataset_cfg):
    """Print the resolved settings and try one listing call. Returns True on success.
    Used by `python download_lakefs_data.py --check`."""
    section = (dataset_cfg or {}).get('lakefs')
    if not section:
        print("[lakeFS] this dataset section has no `lakefs:` block, so local disk is used.")
        return True
    try:
        loader = make_loader(dataset_cfg, verbose=True)
    except LakeFSConfigError as e:
        print(f"[lakeFS] configuration error:\n{e}")
        return False
    if loader is None:
        return True
    try:
        loader.lakefs.list_objects_v2(Bucket=loader.repo_name,
                                      Prefix=loader.branch_id, MaxKeys=1)
    except Exception as e:
        print(f"[lakeFS] could not reach the repository: {type(e).__name__}: {e}\n"
              f"  Check the endpoint, the access key and secret, and that the repository\n"
              f"  '{loader.repo_name}' and branch '{loader.branch_id.rstrip('/')}' exist.")
        return False
    print(f"[lakeFS] connection OK, repository '{loader.repo_name}' is reachable.")
    return True
