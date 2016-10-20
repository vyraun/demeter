class BaseEnvironment(object):
    def __init__(self, name):
        self.name = name

    def reset(self):
        pass

    def step(self, action):
        pass
