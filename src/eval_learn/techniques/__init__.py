from .sld.wrapper import SLDWrapper
from .uce.wrapper import UCEWrapper
from .concept_steerers.wrapper import ConceptSteerersWrapper
from .free_run.wrapper import FreeRunTechnique
from .mace.wrapper import MACEWrapper

try:
    from .saeuron.wrapper import SAeUronWrapper
except Exception as e:
    import logging
    logging.getLogger(__name__).warning("Could not register saeuron: %s", e)

try:
    from .esd.wrapper import ESDWrapper
except Exception as e:
    import logging
    logging.getLogger(__name__).warning("Could not register esd: %s", e)

try:
    from .SAFREE.wrapper import SAFREETechnique
except Exception as e:
    import logging
    logging.getLogger(__name__).warning("Could not register safree: %s", e)

try:
    from .advunlearn.wrapper import AdvUnlearnWrapper
except Exception as e:
    import logging
    logging.getLogger(__name__).warning("Could not register advunlearn: %s", e)

try:
<<<<<<< src/eval_learn/techniques/__init__.py
    from .ca.wrapper import CAWrapper
except Exception as e:
    import logging
    logging.getLogger(__name__).warning("Could not register ca: %s", e)

__all__ = ["SLDWrapper", "ConceptSteerersWrapper", "UCEWrapper", "FreeRunTechnique", "MACEWrapper", "CAWrapper"]
=======
    from .cogfd.wrapper import CoGFDWrapper
except Exception as e:
    import logging
    logging.getLogger(__name__).warning("Could not register cogfd: %s", e)

try:
    from .ssd.wrapper import SSDWrapper
except Exception as e:
    import logging
    logging.getLogger(__name__).warning("Could not register ssd: %s", e)

__all__ = [
    "SLDWrapper",
    "UCEWrapper",
    "ConceptSteerersWrapper",
    "FreeRunTechnique",
    "MACEWrapper",
    "SAeUronWrapper",
    "ESDWrapper",
    "SAFREETechnique",
    "AdvUnlearnWrapper",
    "CoGFDWrapper",
    "SSDWrapper",
    "CAWrapper",
]
>>>>>>> src/eval_learn/techniques/__init__.py
