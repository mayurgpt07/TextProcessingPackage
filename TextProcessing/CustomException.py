class CustomException(Exception):
    def __init__(self, *args):
        if args:
            self.message = args[0]
        else:
            self.message = "Unknown Error captured"

    def __str__(self):
        if self.message is None or len(self.message) == 0:
            return "Unknown Error captured"
        else:
            return "An error was found: {0}".format(self.message)

            