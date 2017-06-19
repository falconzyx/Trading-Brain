

class Brain(object):
    """Abstract class for a brain used by an agent. Implementations can be
    agent or environment specific, depending on action
    """

    def train(self, X, y, w=None, *args):
        """Training function.

        Args:
            X (numpy.array): input array
            y (numpy.array): target array
            w (numpy.array): sample weights to be used in the fitting process.

        Returns:
            float: value of the loss
        """
        raise NotImplementedError()

    def predict(self, X):
        """Predict function.

        Args:
            X (numpy.array): input array

        Returns:
            numpy.array: prediction array
        """
        raise NotImplementedError()
