def count_lines(filename):
    """Returns the number of lines in a given file"""
    with open(filename) as f:
        return sum(1 for line in f)