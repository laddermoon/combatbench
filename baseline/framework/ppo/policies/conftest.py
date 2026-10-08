"""pytest collection config for policies/.

``todo/`` is a scratch/design parking lot (see todo/DESIGN_OVERVIEW.md):
it holds design docs and partial implementations of policy families that
are not part of the maintained suite.  Its test_*.py files import modules
at their pre-parking paths and reference unimplemented families, so they
must not be collected.
"""

collect_ignore = ["todo"]
