def contains(node, val):
    # where node is a nesded dictionary
    if type(node) == dict:
        for key in node:
            if contains(node[key], val):
                return True
    else:
        if node == val:
            return True


triple_nested_dict = {
    'outer_key1': {
        'inner_key1': {
            'deep_key1': 'value1',
            'deep_key2': 'value2'
        },
        'inner_key2': {
            'deep_key3': 'value3',
            'deep_key4': 'value4'
        }
    },
    'outer_key2': {
        'inner_key3': {
            'deep_key5': 'value5',
            'deep_key6': 'value6'
        },
        'inner_key4': {
            'deep_key7': 'value7',
            'deep_key8': 'value8'
        }
    }
}


print(contains(triple_nested_dict, 'value8'))
