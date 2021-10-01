import pytest
import copy

import numpy as np

from mcalf.utils.collections import DefaultParameter


def test_default_parameter():

    apples = DefaultParameter('apples')
    oranges = DefaultParameter('oranges', 12)
    DefaultParameter('bananas', value=15)
    DefaultParameter(name='grapes', value=30)

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

    assert DefaultParameter('a', 1) == DefaultParameter('b', 1)
    assert DefaultParameter('a', 1) + 0 == DefaultParameter('b', 1) + 0

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
    a = DefaultParameter('A', value=1)
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
    a = DefaultParameter('a', np.arange(10))
    assert np.array_equal((a + 1).eval(), np.arange(10) + 1)
    with pytest.raises(SyntaxError):
        np.array_equal((a + 2 * np.arange(10)).eval(), 3 * np.arange(10))
