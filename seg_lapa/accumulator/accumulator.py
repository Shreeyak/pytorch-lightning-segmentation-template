class LossAccumulator:
    def __init__(self):
        """Utility func to aggregate float/int values and average at end of a run"""
        self.sum = 0
        self.counter = 0

    def accumulate(self, new_val, counts):
        self.sum += new_val
        self.counter += counts

    def get_avg(self):
        return self.sum / self.counter

    def reset(self):
        self.sum = 0
        self.counter = 0
