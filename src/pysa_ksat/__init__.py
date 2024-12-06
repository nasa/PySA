
class KSatAdvancedException(Exception):
    def __init__(self):
        super().__init__('"PySA solvers are required for advanced K-SAT generators (pip install pysa_ksat[advanced])"')
