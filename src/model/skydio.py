import numpy as np

from .model import Model
from ..control_assignment import XControlAssignment


class Skydio(Model):
    def __init__(self) -> None:
        super().__init__()

        self._m = 1.325
        self._I = np.diag([0.06071129, 0.0364684, 0.0254117])
        self._control_assignment = XControlAssignment(0.14, 0.18, 1.0, 0.0201)
