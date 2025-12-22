from .preparation import DatasetSimple, DatasetDetections
from .gait import (
    CasiaBPose,iLIDS,MARS
)


def dataset_factory(name):
    if name == "casia-b":
        return CasiaBPose
    if name=="iLIDS":
        return iLIDS
    if name=="mars":
        return MARS
    raise ValueError()
