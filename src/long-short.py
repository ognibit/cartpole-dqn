"""
Curriculum: long short
"""

from base import main, CONFIG
from commons import PoleLengthCurriculum
from random import uniform

class LongPoleLength(PoleLengthCurriculum):

    def set_pole_length(self, env, steps_tot: int) -> float:
        pole_len: float = 0.5 # in case of longer trainig
        if steps_tot < 40_000:
            pole_len = 1.55
        elif steps_tot < 80_000:
            pole_len = 1.1
        elif steps_tot < 120_000:
            pole_len = 0.65

        env.unwrapped.length = pole_len
        return pole_len

if __name__ == '__main__':
    pc = LongPoleLength()
    CONFIG["checkpoint"] = "long-dqn"
    main(pc)

