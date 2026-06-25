"""Normalization protocol implementations.

Importing this package registers all protocol instances in the global registry.
"""

# Import all protocol modules to trigger registration
from science.dtie.normalize.protocols import graph_default  # noqa: F401
from science.dtie.normalize.protocols import family_compare  # noqa: F401
from science.dtie.normalize.protocols import binding_site  # noqa: F401
from science.dtie.normalize.protocols import interface  # noqa: F401
