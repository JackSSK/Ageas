#!/usr/bin/env python3
"""Hangar object for Ageas.

The hangar stores all candidate models and their associated configurations
so that they can later be selected and dispatched into a sortie squad for
training, evaluation, and feature extraction.

TODO:
    - Add methods to add/remove units from the hangar.
    - Add methods to hybridize models.
"""
import os
import logging

import ageas.tool.JSON as JSON
from ageas.unit import Unit

_logger = logging.getLogger(__name__)


class Hangar:
    """Collection of candidate :class:`~ageas.Unit` models and configs.

    Each subdirectory under the configuration folder is treated as a model
    type (e.g. ``svc``, ``mnb``, ``xgb``), and every JSON file in that
    subdirectory yields one :class:`~ageas.Unit` registered in the hangar
    under a unique key.

    :ivar units: Registry mapping ``unit_id`` strings to :class:`~ageas.Unit`
        objects loaded from the config folder.
    """

    def __init__(self, config_folder: str) -> None:
        """Initialize the hangar from a config folder.

        :param config_folder: Folder path that contains one subfolder per model
            type. Each subfolder must hold one or more JSON configuration
            files, each of which becomes a :class:`~ageas.Unit`.
        """
        self.units: dict = {}
        # Sorted for a reproducible unit order; hidden entries (.DS_Store,
        # sync-client files, .ipynb_checkpoints) are not configs.
        for model_type in sorted(os.listdir(config_folder)):
            type_dir = os.path.join(config_folder, model_type)
            if model_type.startswith('.') or not os.path.isdir(type_dir):
                continue
            for unit_file in sorted(os.listdir(type_dir)):
                if unit_file.startswith('.'):
                    continue
                # splitext keeps dots inside names such as 'logreg_C_0.1'.
                tail = os.path.splitext(unit_file)[0]
                unit_id = f"{model_type}_{tail}"
                self.units[unit_id] = Unit(
                    tail=tail,
                    unit_type=model_type,
                    config=JSON.decode(os.path.join(type_dir, unit_file)),
                )
        _logger.debug("Hangar loaded %d units from %s", len(self.units), config_folder)

    def sortie_generate(
        self,
        unit_types: list = None,
        unit_list: list = None,
    ) -> dict:
        """Generate a sortie squad of units.

        A sortie squad is a (possibly filtered) shallow copy of the hangar's
        unit registry which is later passed to :class:`~ageas.Deck` for
        training and evaluation.

        :param unit_types: List of model types to keep (e.g.
            ``['svc', 'mnb', 'xgb']``). ``None`` keeps all types.
        :param unit_list: Whitelist of explicit unit IDs to keep. ``None``
            keeps all units.
        :returns: Mapping ``unit_id -> Unit`` for the units that survived the
            filter.
        """
        squad = self.units.copy()
        if unit_types is not None:
            squad = {k: v for k, v in squad.items() if v.type in unit_types}
        if unit_list is not None:
            squad = {k: v for k, v in squad.items() if k in unit_list}
        return squad

    def spot_check(self, unit_id: str) -> str:
        """Resolve the registry spot of a unit by its public unit ID.

        :param unit_id: Unit identifier of the form ``"{type}_{tail}"`` (e.g.
            ``'svc_unit1'``, ``'mnb_unit2'``).
        :returns: The hangar key that points to the matching unit, or ``None``
            if no unit in the hangar has the requested ID.
        """
        for spot, unit in self.units.items():
            if unit.id == unit_id:
                return spot
        return None
