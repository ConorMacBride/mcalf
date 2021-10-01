import collections
import copy


__all__ = ['DefaultParameter']


class DefaultParameter:
    """
    A named parameter with a optional default value.

    The default value can be changed at any time and very basic
    operations can be queued and evaluated on demand.

    Parameters
    ----------
    name : str
        Name of parameter.
    value : int or float, optional, default=None
        Value of parameter. Other types may work but not supported.

    Attributes
    ----------
    name : str
        Name of parameter.
    value : int or float
        Value of parameter, without operations applied.
    operations : list
        List of operations to apply.

    Raises
    ------
    ValueError
        The parameter is evaluated without a value.
    NotImplementedError
        Operations which require parentheses are applied.

    See Also
    --------
    DefaultParameters : Collection of synced DefaultParameter objects.

    Notes
    -----
    Basic operations can be applied to this object, i.e., ``+``, ``-``, ``*`` and ``/``.
    The ``DefaultParameter`` object must be the left-most entity in an expression.
    Expressions which require parentheses are not supported.
    The only types that are supported in expressions are ``DefaultParameter``, ``float``
    and ``int``.
    Any type should work as a value for ``DefaultParameter`` (such as an array), however,
    only ``float`` and ``int`` are supported.
    See the examples for more details in how it works.

    Examples
    --------
    >>> DefaultParameter('x')
    'x'
    >>> DefaultParameter('x', 12)
    '12'
    >>> DefaultParameter('x') + 1
    'x+1'
    >>> DefaultParameter('x', 1) + 1
    'x+1'
    >>> a = DefaultParameter('y', 10)
    >>> a
    '10'
    >>> a == 10
    True
    >>> a.eval()
    10
    >>> (a * 10).eval()
    100
    >>> a * 10 + 1.5
    'y*10+1.5'
    >>> (a + 10) * 2
    Traceback (most recent call last):
     ...
    NotImplementedError: DefaultParameter equations with brackets are not supported.
    """
    def __init__(self, name, value=None):
        self.name = str(name)
        self.value = value
        self.operations = []

    def __add__(self, other):
        return self.apply_operation('+', other)

    def __sub__(self, other):
        return self.apply_operation('-', other)

    def __mul__(self, other):
        return self.apply_operation('*', other)

    def __truediv__(self, other):
        return self.apply_operation('/', other)

    def apply_operation(self, op, other):
        """
        Apply an operation to the ``DefaultParameter``.

        Parameters
        ----------
        op : str
            Operation symbol to apply. ``+``, ``-``, ``*`` or ``/``.
        other : float or int
            Number on RHS.

        Returns
        -------
        obj
            A copy of ``self`` with the operation added to the queue.

        Notes
        -----
        You should only use this method directly if a builtin Python operation
        is not otherwise implemented in this class, i.e., do ``a + 1`` and not
        ``a.apply_operation('+', 1)``.
        """
        obj = self.copy()  # Operate on a copy
        obj.operations += [(op, other)]  # Add to queue
        obj.verify()  # Verify no parentheses
        return obj

    def __eq__(self, other):
        # Test equality against the expression's numeric value.
        # If the parameter doesn't have a default value,
        # test equality against the string representation
        # of the unevaluated expression.
        try:
            exp = self.eval()
        except ValueError:
            exp = str(self)
        return exp == other

    @property
    def value_or_name(self):
        if self.value is None:
            return self.name
        else:
            return self.value

    def eval(self):
        """Evaluate the expression using the parameter's current value."""
        if self.value is None:
            raise ValueError('Cannot evaluate without a value.')
        return eval(
            str(self),
            {'__builtins__': None},
            {self.name: self.value}
        )

    def __str__(self):
        if len(self.operations) == 0:
            return str(self.value_or_name)
        ret = str(self.name)
        for op, val in self.operations:
            if isinstance(val, DefaultParameter):
                val = val.eval()
            ret += f'{op}{val}'
        return ret

    def __repr__(self):
        return f"'{str(self)}'"

    def verify(self):
        """Enforce order of operations."""
        disable_mul_div = False
        for op, _ in self.operations:
            if op in ('+', '-'):
                disable_mul_div = True
            elif op in ('*', '/') and disable_mul_div:
                raise NotImplementedError(
                    'DefaultParameter equations with brackets are not supported.'
                )

    def copy(self):
        """Return a copy of the parameter."""
        return self.__copy__()

    def __copy__(self):
        obj = self.__class__(copy.copy(self.name), copy.copy(self.value))
        obj.operations = copy.deepcopy(self.operations)
        return obj

    def __deepcopy__(self, memodict={}):
        return self.__copy__()
