def load_css(files):
    css = ""
    for file in files:

        with open(file) as f:
            css += f.read()

    return css