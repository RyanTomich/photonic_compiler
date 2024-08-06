import testing as test
import re
import importlib

# Code format

template = "{start_time:.10f} {hardware:<6} {core:4d}   {opp:<10} {func:<50}\n"

def get_class_obj(hardware_core):
    if '<' not in hardware_core:
        return hardware_core, 0

    match = re.search(r">(\d+)$", hardware_core)
    if match:
        core = int(match.group(1))
    else:
        raise ValueError("Invalid object string format")

    match = re.match(r"<hardware.(.*) object at", hardware_core)
    if match:
        class_name = match.group(1)
    else:
        raise ValueError("Invalid object string format")

    return class_name, core

def code_gen(scheduled_flat_graph):
    sorted_nodes = scheduled_flat_graph.get_sorted_nodes()
    with open("code.txt", "w") as file:
        for node in sorted_nodes:
            hardware, core = get_class_obj(node.hardware_selection)
            file.write(
                template.format(
                    start_time=node.start_time,
                    hardware=hardware,
                    core=core,
                    opp=node.stack.opp,
                    func=node.stack.tvm_func if node.stack.tvm_func is not None else 'None'
                )
            )

    schedule_df = scheduled_flat_graph.create_schedule_data(write=True)
    stagnent_time = test.schedule_validate(schedule_df)
