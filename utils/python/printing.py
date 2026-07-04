
class Printing: 
    indent  = ""
    verbose = True

    def set_indent(self, indent): 
        self.indent = indent 
    
    def set_verbosity(self, verbose): 
        self.verbose = verbose

    def print(self, *args, **kwargs): 
        if self.verbose == True:
            print(self.indent + args[0], *args[1:], **kwargs)
