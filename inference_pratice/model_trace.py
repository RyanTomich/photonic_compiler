import os

# Create Write File
current_directory = os.path.dirname(os.path.abspath(__file__))
output_file_path = os.path.join(current_directory, 'GPT2_instructions.txt')
instrucionts_txt = open(output_file_path, "w")


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
        instrucionts_txt.write(f"{func_name}\n")
        result = func(*args, **kwargs)
        return result
    return wrapper
