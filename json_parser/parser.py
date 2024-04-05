import json

with open('/home/rjtomich/photonic_compiler/Pytorch-LeNet/LeNet_graph.json') as json_file:
    data = json.load(json_file) # returns json file as dict

parsed_txt = open("parsed.txt", "w") # creates the write file in write mode append ('a') mode also exists


# Nicer visualization
if "nodes" in data:
    for num, node in enumerate(data['nodes']):
        parsed_txt.write(f"nodes[{num}]:\n")
        for key in node:
            if key == "attrs":
                parsed_txt.write(f"         {key}: \n")
                for attribute in node[key]:
                    parsed_txt.write(f"           {attribute}: {node[key][attribute]}\n")
                continue
            parsed_txt.write(f"         {key}: {node[key]}\n")


        # print(f"nodes[{num}]: {node}")
        # print("")

if "attrs" in data:
    attrs = data["attrs"]
    if "shape" in attrs:
        parsed_txt.write(f"Shape attributes: {attrs['shape'][1]}")
