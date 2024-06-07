import os

# Create Write File
current_directory = os.path.dirname(os.path.abspath(__file__))
output_file_path = os.path.join(current_directory, 'GPT2_instructions.txt')
instrucionts_txt = open(output_file_path, "w")
func_calls = {}
dependancy_graph = {'START': None}


def for_all_methods(decorator):
    def decorate(cls):
        for attr in cls.__dict__: # there's propably a better way to do this
            if callable(getattr(cls, attr)):
                setattr(cls, attr, decorator(getattr(cls, attr)))
        return cls
    return decorate


def file_write(name, details=''):
    file_write.calls += 1
    func_calls.setdefault(name, 0)
    node = DependancyNode(name, file_write.calls, None, details, 'CPU')
    if name == 'row_softamx':
        file_write.para.append(node)
        dependancy_graph[file_write.prev].append(node)
    else:
        if file_write.para:
            for para_node in file_write.para:
                dependancy_graph[para_node] = [node]
            file_write.para = []
        else:
            dependancy_graph[file_write.prev] = [node]
        dependancy_graph[node] = []
        file_write.prev = node
    func_calls[name] += 1
    instrucionts_txt.write(f'({file_write.calls}){name}:{details}\n')

file_write.calls = 0
file_write.prev = 'START'
file_write.para = []

def catch_name(func):
    def wrapper(*args, **kwargs):
        func_name = str(func.__name__)
        file_write(f"{func_name}")
        result = func(*args, **kwargs)
        return result
    return wrapper

class DependancyNode():
    def __init__(self, name, oppid, opp, size, hardware):
        self.name = name
        self.oppid = oppid
        self.opp = opp
        self.size = size
        self.hardware = hardware

    def __hash__(self):
        return hash(self.oppid)

    def __eq__(self, other):
        if isinstance(other, DependancyNode):
            return self.oppid == other.oppid
        return False

    def __str__(self):
        return f'{self.name=} \n{self.oppid=} \n{self.opp=} \n{self.size=} \n{self.hardware=}'
