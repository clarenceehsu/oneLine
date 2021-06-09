"""
An error class for raising different errors with clear information.
"""


class Error(object):

    def __init__(self):
        return

    def _input_error(self):
        """
        The exception of ValueError when format was unsupported.

        :return: ValueError
        """

        raise ValueError('Input error, please input a valid data that satisfied.')

    def _raise_parameter_error(self, args):
        """
        The exception of ValueError when format was unsupported.

        :return: ValueError
        """

        raise ValueError('Input parameter {} {} not exist.'.format(", ".join(args), "does" if len(args) == 1 else "do"))

    def _wrong_parameter_error(self, args):
        """
        The exception of ValueError when the input parameters were wrong.

        :return: ValueError
        """

        raise ValueError('The input parameter {} {} the wrong value.'.format(", ".join(args), "has" if len(args) == 1 else "have"))

    def _raise_plot_value_error(self, s: list):
        """
        A ValueError would be raised if required value was missed.

        :param s: the missing value(s)
        :return: None
        """

        raise ValueError(f'The parameter { ", ".join(s) } {"is" if len(s) == 1 else "are"} required.')

    def _raise_plot_format_error(self, s: list, format: str):
        """
        A ValueError of format would be raised if the format was not matched.

        :param s: the wrong parameters
        :param format: the required format
        :return: None
        """

        raise ValueError(f'The parameter { ", ".join(s) } should be { format}, rather than { ", ".join([str(type(n)) for n in s]) }.')
