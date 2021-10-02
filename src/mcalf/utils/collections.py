import collections
import copy


__all__ = ['Parameter', 'ParameterDict', 'OrderedParameterDict']


class Parameter:
    """
    A named parameter with a optional value.

    The value can be changed at any time and very basic
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
    Parameters : Collection of synced ``Parameter`` objects.

    Notes
    -----
    Basic operations can be applied to this object, i.e., ``+``, ``-``, ``*`` and ``/``.
    The ``Parameter`` object must be the left-most entity in an expression.
    Expressions which require parentheses are not supported.
    The only types that are supported in expressions are ``Parameter``, ``float``
    and ``int``.
    Any type should work as a value for ``Parameter`` (such as an array), however,
    only ``float`` and ``int`` are supported.
    See the examples for more details in how it works.

    Examples
    --------
    >>> Parameter('x')
    'x'
    >>> Parameter('x', 12)
    '12'
    >>> Parameter('x') + 1
    'x+1'
    >>> Parameter('x', 1) + 1
    'x+1'
    >>> a = Parameter('y', 10)
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
    NotImplementedError: Parameter equations with brackets are not supported.
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
        Apply an operation to the ``Parameter``.

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
        # If the parameter doesn't have a value,
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
            if isinstance(val, Parameter):
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
                    'Parameter equations with brackets are not supported.'
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


class SyncedParameters:

    @property
    def _tracked(self):
        if not hasattr(self, '_tracked_objs'):
            self._tracked_objs = {}
        return self._tracked_objs

    def _track_object(self, obj):

        if obj.value is not None and self.exists(obj.name):

            # No more than one value should be associated with each name
            if self.has_value(obj.name) and obj.value is not self.get_parameter(obj.name):
                raise ValueError(f"'{obj.name}' value {obj.value} does not match "
                                 f"existing value {self.get_parameter(obj.name)}.")

            # Sync the value
            self.update_parameter(obj.name, obj.value)

        if self.exists(obj.name):
            self._tracked[obj.name]['objects'] += [obj]
        else:
            self._tracked[obj.name] = {'value': obj.value, 'objects': [obj]}

    def update_parameter(self, name, value):
        self._tracked[name]['value'] = value
        for obj in self._tracked[name]['objects']:
            obj.value = value

    def get_parameter(self, name):
        return self._tracked[name]['value']

    def has_value(self, name):
        return False if self.get_parameter(name) is None else True

    def exists(self, name):
        try:
            self._tracked[name]
        except KeyError:
            return False
        else:
            return True


class BaseParameterDict(SyncedParameters):
    """
    A base class for dictionaries of `Parameter` objects.

    The same parameters existing across multiple dictionary
    values can be kept in sync with each other.
    """
    def __setitem__(self, key, value):
        if isinstance(value, Parameter):
            self._track_object(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, Parameter):
                    self._track_object(item)
        super().__setitem__(key, value)

    def eval(self):
        edict = self.__class__.__bases__[-1]()
        for key, value in self.items():
            if isinstance(value, Parameter):
                value = value.eval()
            elif isinstance(value, list):
                value = []
                for i, v in enumerate(value):
                    if isinstance(v, Parameter):
                        value += [v.eval()]
                    else:
                        value += [v]
            edict[key] = value
        return edict


class ParameterDict(BaseParameterDict, collections.UserDict):
    """
    An unordered dictionary of `Parameter` objects.

    The same parameters existing across multiple dictionary
    values can be kept in sync with each other.
    """


class OrderedParameterDict(BaseParameterDict, collections.OrderedDict):
    """
    An ordered dictionary of `Parameter` objects.

    The same parameters existing across multiple dictionary
    values can be kept in sync with each other.
    """
