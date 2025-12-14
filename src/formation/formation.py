import abc
import numpy as np


class Formation(abc.ABC):       #定义编队接口

    def __init__(self, delta: float, count: int) -> None:
        super().__init__()
        self._delta = delta       #编队中无人机的间距参数
        self._count = count         #编队中无人机数量

    def cal_deltas(self, psi: float):
        #初始化所有无人机的相对位置偏差为0
        deltas = [np.zeros((3, self._count)) for _ in range(self._count)]
        deltas[0] = self.cal_delta0(psi)#第0架领航机由子类实现
        #计算其余无人机的相对偏差
        for i in range(1, self._count):
            for j in range(self._count):
                if j < i:
                    deltas[i][:, j] = -deltas[j][:, i]
                elif j > i:
                    deltas[i][:, j] = deltas[i - 1][:, j] + deltas[i][:, i - 1]

        return deltas

    @abc.abstractmethod
    def cal_delta0(self, psi: float) -> np.ndarray:
        #抽象方法，计算第0架无人机相对于其他无人机的初始偏差（由子类实现具体编队形状）
        pass
