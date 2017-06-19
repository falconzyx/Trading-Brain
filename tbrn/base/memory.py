class Memory(object):
    """Abstract class for a memory used by an agent.
    """
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.i = 0
        
    def add(self, state, action, reward, next_state, terminal):
        """Add a transition.
        """
        self.i = (self.i+1) % self.memory_size

    def sample(self, size):
        """Sample transitions

        Returns:
            list: List of sampled transitions
        """
        raise NotImplementedError()

    def getIndex(self):
        """Get current index
        
        Returns:
            int : current Index
        """
        return self.i