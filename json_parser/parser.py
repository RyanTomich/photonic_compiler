import json

with open('/home/rjtomich/photonic_compiler/Pytorch-LeNet/LeNet_graph.json') as json_file:
    raw_json = json.load(json_file) # returns json file as dict

parsed_txt = open("parsed.txt", "w") # creates the write file in write mode append ('a') mode also exists


# # Nicer visualization
# if "nodes" in raw_json:
#     for num, node in enumerate(raw_json['nodes']):
#         parsed_txt.write(f"nodes[{num}]:\n")
#         for key in node:
#             if key == "attrs":
#                 parsed_txt.write(f"         {key}: \n")
#                 for attribute in node[key]:
#                     parsed_txt.write(f"           {attribute}: {node[key][attribute]}\n")
#                 continue
#             parsed_txt.write(f"         {key}: {node[key]}\n")


#         # print(f"nodes[{num}]: {node}")
#         # print("")

# if "attrs" in raw_jsopn:
#     attrs = raw_json["attrs"]
#     if "shape" in attrs:
#         parsed_txt.write(f"Shape attributes: {attrs['shape'][1]}")

def contains(node, val):
    # where node is a nesded dictionary
    if type(node) == dict:
        for key in node:
            if contains(node[key], val):
                return True
    else:
        if val in node:
            return True

def get_shape_index(node):
    return[input[0] for input in node["inputs"]]


for order, node in enumerate(raw_json["nodes"]):
    if contains(node, 'null'):
        parsed_txt.write(f"[null]    {raw_json['nodes'][order]}\n")
    elif contains(node, 'dense'):
        parsed_txt.write(f"[relu/MAC]{raw_json['nodes'][order]}\n")
        parsed_txt.write(f"     input shapes:")
        for index in get_shape_index(node):
            parsed_txt.write(f"{str(raw_json['attrs']['shape'][1][index])}")
        parsed_txt.write("\n")

    else:
        parsed_txt.write(f"[]    {raw_json['nodes'][order]}\n")
