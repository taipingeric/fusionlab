import inspect

# Define the show_classtree function
def show_classtree(clss, indent=0):
    # Get the full argument spec for the class
    argspec = inspect.getfullargspec(clss)
    # Get the arguments for the class
    args = argspec.args
    # If the class has a varargs argument, append it to args
    if argspec.varargs:
        args.append('*' + argspec.varargs)
    # If the class has a varkw argument, append it to args
    if argspec.varkw:
        args.append('**' + argspec.varkw)
    # Print the class name and arguments, indented by indent spaces
    print('  ' * indent + f'{clss} | input: {args}')
    # For each base class of the class
    for supercls in clss.__bases__:
        # Recursively call show_classtree on the base class, with an indent of 3 more spaces
        show_classtree(supercls, indent + 3)