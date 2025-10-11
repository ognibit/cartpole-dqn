"""
Curriculum: uniform
"""

from base import main, CONFIG
from commons import PoleLengthCurriculum
from random import uniform

class UniformPoleLength(PoleLengthCurriculum):

    def set_pole_length(self, env, steps_tot: int) -> float:
        pole_len: float = 1.1
        env.unwrapped.length = pole_len
        return pole_len

if __name__ == '__main__':
    pc = UniformPoleLength()
    CONFIG["checkpoint"] = "uniform-dqn"
    main(pc)
