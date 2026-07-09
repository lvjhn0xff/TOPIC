
class Printing: 
    indent  = 0
    verbose = True

    def set_indent(self, indent): 
        self.indent = indent 
    
    def set_verbosity(self, verbose): 
        self.verbose = verbose

    def print(self, *args, **kwargs): 
        if self.verbose == True:
            indentation = "\t" * self.indent
            print(indentation + args[0], *args[1:], **kwargs)
