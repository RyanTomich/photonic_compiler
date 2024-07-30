class Hardware:
    def __init__(self, clock_speed):
        self.clock_speed = clock_speed

class PhotonicHardware(Hardware):
    def __init__(self, clock_speed, num_cores, num_multiplex):
        super().__init__(clock_speed)
        self.num_numtiplex = num_multiplex
        self.num_cores = num_cores

class CPU(Hardware):
    def __init__(self, clock_speed, num_cores):
        super().__init__(clock_speed)
        self.num_cores = num_cores

class GPU(Hardware):
    def __init__(self, clock_speed):
        super().__init__(clock_speed)
