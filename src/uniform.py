"""
Curriculum: uniform
"""

from base import main, CONFIG
from commons import PoleLengthCurriculum

class UniformPoleLength(PoleLengthCurriculum):

    def set_pole_length(self, env, episode: int, esteps: int, tsteps: int) -> float:
        #FIXME implement
        pole_len: float = env.unwrapped.length
        return pole_len

if __name__ == '__main__':
    pc = UniformPoleLength()
    CONFIG["checkpoint"] = "uniform-dqn"
    main(pc)
