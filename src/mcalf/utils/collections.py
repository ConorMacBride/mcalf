import copy
import collections

__all__ = ['Parameter', 'ParameterDict', 'OrderedParameterDict',
           'BaseParameterDict', 'SyncedParameters']


class Parameter:
    """A named parameter with a optional value.

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
    >>> a = Parameter('y') * 2
    >>> a
    'y*2'
    >>> a.value = 5
    >>> a == 10
    True
    >>> a.eval()
    10
    >>> (a * 10).eval()
    100
    >>> a * 10 + 1.5
    'y*2*10+1.5'
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
        """Apply an operation to the ``Parameter``.

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
        """Returns its value if set, otherwise returns its name."""
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
    """Database for keeping :class:`Parameter` objects in sync.

    :class:`Parameter` objects with the same name can be linked with this class
    and kept in sync with each other.

    Examples
    --------
    >>> a = Parameter('x') + 1
    >>> b = Parameter('x', 2) * 3
    >>> c = Parameter('y')

    Now that :class:`Parameter` objects have been created,
    we can keep them in sync.

    >>> s = SyncedParameters()
    >>> s.track_object(a)
    >>> s.track_object(b)
    >>> s.track_object(c)

    Notice how the value of `a` has been synced to match the
    value provided in `b`.

    >>> a == 3
    True

    The value of a named parameter can be updated and the
    change is propagated to all :class:`Parameter` objects
    of that name.

    >>> s.update_parameter('x', 3)
    >>> a == 4
    True
    >>> b == 9
    True
    """
    @property
    def _tracked(self):
        """Dictionary of tracked :class:`Parameters` grouped by name.

        Notes
        -----
        Has the form: ``{Parameter.name: {'value': Parameter.value, 'objects': List[Parameter]}}``.
        All ``Parameter.value`` of the same ``Parameter.name`` are
        identical in both ``'value'`` and all in ``'objects'``.
        """
        if not hasattr(self, '_tracked_objs'):
            self._tracked_objs = {}
        return self._tracked_objs

    def track_object(self, obj):
        """Add a new :class:`Parameter` object to keep in sync.

        Parameters
        ----------
        obj : :class:`Parameter`
            The parameter object to keep in sync.

        Notes
        -----
        Any objects added must not have a value that contradicts the existing
        value of the parameter (if set). Objects added which have no value set
        will inherit the value of the other parameters of the same name.
        """
        if self.exists(obj.name):  # If parameter already registered...
            if obj.value is not None:  # ...copy incoming value to existing.
                # (No more than one value should be associated with each name.)
                if self.has_value(obj.name) and obj.value is not self.get_parameter(obj.name):
                    raise ValueError(f"'{obj.name}' value {obj.value} does not match "
                                     f"existing value {self.get_parameter(obj.name)}.")
                self.update_parameter(obj.name, obj.value)
            elif self.has_value(obj.name):  # ...copy existing value to incoming.
                obj.value = self.get_parameter(obj.name)
            self._tracked[obj.name]['objects'] += [obj]
        else:
            self._tracked[obj.name] = {'value': obj.value, 'objects': [obj]}

    def update_parameter(self, name, value):
        """Update the value of a named parameter.

        All tracked :class:`Parameter` objects of this name will be updated.

        Parameters
        ----------
        name : string
            Name of parameter to update.
        value : float or int
            New value of parameter. Other types may work.
        """
        self._tracked[name]['value'] = value
        for obj in self._tracked[name]['objects']:
            obj.value = value

    def get_parameter(self, name):
        """Get the value of the named parameter.

        Parameters
        ----------
        name : str
            Name of parameter to get value of.

        Returns
        -------
        value
            The current value of the parameter. ``None`` if not set.
        """
        return self._tracked[name]['value']

    def has_value(self, name: str) -> bool:
        """Whether the named parameter has a value set."""
        return False if self.get_parameter(name) is None else True

    def exists(self, name: str) -> bool:
        """Whether any parameters of a particular name are tracked."""
        try:
            self._tracked[name]
        except KeyError:
            return False
        else:
            return True


class BaseParameterDict(SyncedParameters):
    """
    A base class for dictionaries of :class:`Parameter` objects.

    The same parameters existing across multiple dictionary
    values can be kept in sync with each other.
    For a :class:`Parameter` object to be kept in sync it must be
    located in the dictionary such that ``{key: Parameter}`` or
    ``{key: List[Parameter]}``.
    """
    def __setitem__(self, key, value):
        if isinstance(value, Parameter):
            self.track_object(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, Parameter):
                    self.track_object(item)
        super().__setitem__(key, value)

    def eval(self):
        """Return a copy of the dictionary with all :class:`Parameter` objects evaluated."""
        edict = self.__class__.__bases__[-1]()
        for key, value in self.items():
            if isinstance(value, Parameter):
                value = value.eval()
            elif isinstance(value, list):
                lst = []
                for i, v in enumerate(value):
                    if isinstance(v, Parameter):
                        lst += [v.eval()]
                    else:
                        lst += [v]
                value = lst
            edict[key] = value
        return edict


class ParameterDict(BaseParameterDict, collections.UserDict):
    """
    An unordered dictionary of :class:`Parameter` objects.

    The same parameters existing across multiple dictionary
    values can be kept in sync with each other.
    For a :class:`Parameter` object to be kept in sync it must be
    located in the dictionary such that ``{key: Parameter}`` or
    ``{key: List[Parameter]}``.

    Examples
    --------
    >>> d = ParameterDict({
    ...     'a': Parameter('x') + 1,
    ...     'b': [2, Parameter('x'), 5],
    ...     'c': {1, 2, 3},
    ... })
    >>> d
    {'a': 'x+1', 'b': [2, 'x', 5], 'c': {1, 2, 3}}
    >>> d.update_parameter('x', 1)
    >>> d.eval()
    {'a': 2, 'b': [2, 1, 5], 'c': {1, 2, 3}}
    """


class OrderedParameterDict(BaseParameterDict, collections.OrderedDict):
    """
    An ordered dictionary of :class:`Parameter` objects.

    The same parameters existing across multiple dictionary
    values can be kept in sync with each other.
    For a :class:`Parameter` object to be kept in sync it must be
    located in the dictionary such that ``{key: Parameter}`` or
    ``{key: List[Parameter]}``.

    Examples
    --------
    >>> d = OrderedParameterDict([
    ...     ('a', Parameter('x') + 1),
    ...     ('b', [2, Parameter('x'), 5]),
    ...     ('c', {1, 2, 3}),
    ... ])
    >>> d
    OrderedParameterDict([('a', 'x+1'), ('b', [2, 'x', 5]), ('c', {1, 2, 3})])
    >>> d.update_parameter('x', 1)
    >>> d.eval()
    OrderedDict([('a', 2), ('b', [2, 1, 5]), ('c', {1, 2, 3})])
    """
