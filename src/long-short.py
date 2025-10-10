"""
Curriculum: long short
"""

from base import main, CONFIG
from commons import PoleLengthCurriculum
from random import uniform

class LongPoleLength(PoleLengthCurriculum):

    def set_pole_length(self, env, episode: int) -> float:
        pole_len: float = 0.5 # in case of longer trainig
        if episode < 100:
            pole_len = uniform(1.31, 1.8)
        elif episode < 200:
            pole_len = uniform(0.9, 1.3)
        elif episode < 300:
            # full range
            pole_len = uniform(0.4, 1.8)

        env.unwrapped.length = pole_len
        return pole_len

if __name__ == '__main__':
    pc = LongPoleLength()
    CONFIG["checkpoint"] = "long-dqn"
    main(pc)

