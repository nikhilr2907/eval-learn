import sys
from .local import (
    register_technique,
    register_metric,
    register_dataset,
    register_benchmark,
)
from ..logging_utils import get_logger

logger = get_logger(__name__)

# Compatibility for Python 3.8/3.9
if sys.version_info < (3, 10):
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points


def load_entrypoints():
    """
    Discover and load plugins from entry points.
    """
    groups = {
        "eval_learn.techniques": register_technique,
        "eval_learn.metrics": register_metric,
        "eval_learn.datasets": register_dataset,
        "eval_learn.benchmarks": register_benchmark,
    }

    logger.info("Loading plugins from entry points...")

    for group_name, decorator in groups.items():
        # In Python 3.10+, entry_points(group=...) returns a generic collection
        try:
            eps = entry_points(group=group_name)
        except TypeError:
            # Fallback for older importlib_metadata or Python < 3.10 behavior if needed
            all_eps = entry_points()
            if hasattr(all_eps, "select"):
                eps = all_eps.select(group=group_name)
            else:
                eps = all_eps.get(group_name, [])

        for ep in eps:
            try:
                logger.debug(f"Loading plugin: {ep.name} from {ep.value}")
                plugin_obj = ep.load()
                # Apply the registration decorator
                # The name in the entry point (ep.name) overrides the name in the code if we want,
                # but usually the decorator handles it.
                # However, our local decorators take a name argument.
                # If the object is already decorated, re-decorating might be redundant but safe if idempotent.
                # If it's NOT decorated, we register it with the entry point name.

                # Check if it's already registered via side-effect of import?
                # Actually, simply loading it might have triggered the decorator if it was used in the file.
                # But if the user exposes a raw class/function without decorator, we register it manually.

                decorator(ep.name)(plugin_obj)
                logger.info(f"Registered plugin '{ep.name}' ({group_name})")
            except Exception as e:
                logger.error(f"Failed to load plugin {ep.name}: {e}")
