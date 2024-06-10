import os
import numpy as np

# Catching functions
def for_all_methods(decorator):
    def decorate(cls):
        for attr in cls.__dict__: # there's propably a better way to do this
            if callable(getattr(cls, attr)):
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls
    return decorate

def catch_name(func):
    def wrapper(*args, **kwargs):
        func_name = str(func.__name__)
        file_write(f"{func_name}")
        result = func(*args, **kwargs)
        return result
    return wrapper


# making dependancy_graph
def file_write(name, opp=None, in_size=()):
    file_write.calls += 1
    node = DependancyNode(name, file_write.calls, opp, in_size, 'CPU')
    if opp in photonic_indicators:
        node.hardware = 'PH'

    if name in parallelizable:
        file_write.para.append(node)
        graph.dependancy_graph[file_write.prev].append(node)
    else:
        if file_write.para:
            for para_node in file_write.para:
                graph.dependancy_graph[para_node] = [node]
            file_write.para = []
        else:
            graph.dependancy_graph[file_write.prev] = [node]
        graph.dependancy_graph[node] = []
        file_write.prev = node
    # instructions_txt.write(f'({file_write.calls}){name}:{details}\n')


class DependancyNode():
    def __init__(self, name, oppid, opp, input_size, hardware):
        self.name = name
        self.oppid = oppid
        self.opp = opp
        self.input_size = input_size
        self.hardware = hardware
        time = self._calc_time(opp, input_size)

    def _calc_time(self, opp, inputs):
        if opp is None:
            return None
        if opp == '@':
            a = np.prod(inputs[0])
            return a*inputs[1][-1]
        if opp in ('+-*/'):
            return np.prod(inputs[0])

    def __hash__(self):
        return hash(self.oppid)

    def __eq__(self, other):
        if isinstance(other, DependancyNode):
            return self.oppid == other.oppid
        return False

    def __str__(self):
        return f'{self.name=} \n{self.oppid=} \n{self.opp=} \n{self.input_size=} \n{self.hardware=}'

class DependancyGraph():
    def __init__(self):
        self.dependancy_graph = {'START': None}

    def print_instructions(self):
        group = self.dependancy_graph["START"]
        while group:
            tab = ''
            if len(group) > 1:
                tab = '   '
            for node in group:
                with open(output_file_path, 'a') as instructions:
                    opp = 'other' if node.opp is None else '  ' + node.opp + '  '
                    instructions.write(f'{node.hardware[0]}: {tab}{opp} {node.name}:{node.input_size}\n')
            group = self.dependancy_graph[node]

    def __str__(self):
        return str(len(self.dependancy_graph))

# Create Write File
current_directory = os.path.dirname(os.path.abspath(__file__))
output_file_path = os.path.join(current_directory, 'GPT2_instructions.txt')
with open(output_file_path, 'w') as instructions:
    pass # clear document

file_write.calls = 0
file_write.prev = 'START'
file_write.para = []
parallelizable = ['row_softamx']
photonic_indicators = ['@']

graph = DependancyGraph()
