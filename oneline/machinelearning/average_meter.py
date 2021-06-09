class AverageMeter(object):
    """
    An average meter for counting and recording the data during process.
    """

    def __init__(self):
        """
        Initialize the meter.
        """

        # the average of the recording parameter
        self.avg = 0
        # the overall sum of the recording parameter
        self.sum = 0
        # the counter of the recording parameter
        self.count = 0

    def reset(self):
        """
        Reset the average meter to the initial.

        :return: None
        """

        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Update the record.

        :param val: the value of parameter
        :param n: number of the value that should be counted
        :return: None
        """

        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
