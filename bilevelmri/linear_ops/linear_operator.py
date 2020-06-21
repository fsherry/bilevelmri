class LinearOperator:
    def __init__(self):
        pass

    def __call__(self, x):
        raise NotImplementedError()

    def T(self, z):
        raise NotImplementedError()
