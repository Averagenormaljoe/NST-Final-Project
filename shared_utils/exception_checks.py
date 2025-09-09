
def none_check(variable,name : str):
    if variable is None:
        raise ValueError(f"Error: {name} cannot be none.")