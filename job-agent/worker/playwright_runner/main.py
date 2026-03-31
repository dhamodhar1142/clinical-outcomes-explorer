from __future__ import annotations

from .portals.generic import GenericApplier
from .portals.greenhouse import GreenhouseApplier
from .portals.lever import LeverApplier
from .portals.workday import WorkdayApplier


APPLIER_MAP = {
    "generic": GenericApplier,
    "workday": WorkdayApplier,
    "greenhouse": GreenhouseApplier,
    "lever": LeverApplier,
}


def get_applier(portal_type: str):
    return APPLIER_MAP.get(portal_type, GenericApplier)
