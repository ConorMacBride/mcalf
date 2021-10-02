import pytest
import copy
import collections

import numpy as np

from mcalf.utils.collections import Parameter, ParameterDict, OrderedParameterDict


def test_parameter():

    apples = Parameter('apples')
    oranges = Parameter('oranges', 12)
    Parameter('bananas', value=15)
    Parameter(name='grapes', value=30)

    # No value set

    assert apples == 'apples'
    assert apples is not None
    assert str(apples) == 'apples'
    assert apples.name == 'apples'
    assert apples.value is None
    assert apples.value_or_name == 'apples'
    assert apples == apples
    assert apples + 1 == apples + 1

    assert apples + 7 == 'apples+7'
    assert apples - 8 == 'apples-8'
    assert apples * 3 == 'apples*3'
    assert apples / 11 == 'apples/11'
    assert apples * 12 + 3 == 'apples*12+3'
    assert apples / 6 - 6 / 2 == 'apples/6-3.0'

    with pytest.raises(ValueError) as e:
        apples.eval()
    assert 'Cannot evaluate without a value.' in str(e.value)

    # Value set

    assert oranges == 12
    assert not oranges == '12'
    assert not oranges == 'oranges'
    assert str(oranges) == '12'
    assert oranges.name == 'oranges'
    assert oranges.value == 12
    assert oranges.value_or_name == 12
    assert oranges == oranges
    assert oranges + 1 == oranges + 1

    assert oranges + 2 == 14
    assert str(oranges + 2) == 'oranges+2'
    assert (oranges * 10 + 4).eval() == 124

    assert Parameter('a', 1) == Parameter('b', 1)
    assert Parameter('a', 1) + 0 == Parameter('b', 1) + 0

    # Equations that need brackets to work are not supported
    with pytest.raises(NotImplementedError):
        (oranges + 2) * 3

    # Parameter must be left-most object
    with pytest.raises(TypeError):
        2 + oranges
    with pytest.raises(TypeError):
        6.0 * oranges
    with pytest.raises(TypeError):
        2 + oranges * 2

    # Copying a parameter
    a = Parameter('A', value=1)
    b = a.copy()
    c = copy.copy(a)
    d = copy.deepcopy(a)
    e = a + 0
    f = a
    a.value = 2
    for p in (b, c, d, e):
        assert p == 1
    assert f == 2

    # Very basic array support
    a = Parameter('a', np.arange(10))
    assert np.array_equal((a + 1).eval(), np.arange(10) + 1)
    with pytest.raises(SyntaxError):
        np.array_equal((a + 2 * np.arange(10)).eval(), 3 * np.arange(10))


def verify_synced_parameters(tracked):
    for p in tracked.values():
        for obj in p['objects']:
            assert obj.value is p['value']


@pytest.mark.parametrize('cls', [OrderedParameterDict, ParameterDict])
def test_parameter_dict(cls):

    # Initialise with parameters
    parameters = [
        ('aa', Parameter('a', 2) + 1),
        ('bb', Parameter('b') + 2),
        ('cc', Parameter('a', 2) + 3),
        ('dd', Parameter('c') + 4),
        ('ee', Parameter('b', 3) + 5),
    ]

    if cls is ParameterDict:
        parameters = {k: v for k, v in parameters}
    x = cls(parameters)

    assert len(x._tracked.keys()) == 3
    assert set(x._tracked.keys()) == {'a', 'b', 'c'}

    verify_synced_parameters(x._tracked)
    assert x['aa'] == 3
    assert x['bb'] == 5
    assert x['cc'] == 5
    assert x['dd'] == 'c+4'
    assert x['ee'] == 8
    assert x.has_value('a')
    assert not x.has_value('c')
    assert x.get_parameter('c') is None

    # Update some parameters
    x.update_parameter('c', 9)
    x.update_parameter('b', None)
    x.update_parameter('a', 12)

    verify_synced_parameters(x._tracked)
    assert x['aa'] == 13
    assert x['bb'] == 'b+2'
    assert x['cc'] == 15
    assert x['dd'] == 13
    assert x['ee'] == 'b+5'

    # Evaluate all parameters
    with pytest.raises(ValueError):
        x.eval()
    x.update_parameter('b', 1)
    ex = x.eval()
    if cls is ParameterDict:
        assert isinstance(ex, collections.UserDict)
    elif cls is OrderedParameterDict:
        assert isinstance(ex, collections.OrderedDict)
    for k, v in [('aa', 13), ('bb', 3), ('cc', 15), ('dd', 13), ('ee', 6)]:
        assert ex[k] is v
        assert not x[k] is v

    assert x.exists('b')
    assert not x.exists('t')

    # Add parameter later
    x['tt'] = Parameter('t', 22) + 5
    assert x.exists('t')
    assert x['tt'] == 27
    assert x.get_parameter('t') == 22
    verify_synced_parameters(x._tracked)
    assert len(x._tracked.keys()) == 4
    assert set(x._tracked.keys()) == {'a', 'b', 'c', 't'}

    # Add standard number
    x['s'] = 10
    assert x['s'] == 10
