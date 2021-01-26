from libc.stddef cimport *
from libc.stdint cimport *
from libc.string cimport memcpy

cimport numpy as np

from qtestpy cimport kdb_api

import datetime
from pathlib import Path
from inspect import signature
import sys
import traceback
from typing import Any, Optional, Type, Union
from uuid import UUID
from uuid import uuid4 as random_uuid

import numpy as np
import pandas as pd
import pyarrow as pa

cdef inline kdb_api.K _k(x):
    return <kdb_api.K><uintptr_t>x._addr


def init(kdb):
    return 0


np_int_types = (
    np.bool_, np.byte, np.ubyte, np.short, np.ushort, np.intc, np.uintc,
    np.int_, np.uint, np.longlong, np.ulonglong, np.int8, np.int16, np.int32,
    np.int64, np.uint8, np.uint16, np.uint32, np.uint64, np.intp, np.uintp
)

np_float_types = (
    np.half, np.float16, np.single, np.double, np.longdouble, np.float32,
    np.float64, np.float_
)

type_number_to_pykdb_k_type = {**k.type_number_to_pykdb_k_type, **{
    98: k.Table,
    99: k.Dictionary,
    101: k.UnaryPrimative,
}}


def resolve_ktype(k_type) -> Optional[Type[k.K]]:
    """Resolve the pykdb.k.K type represented by the k_type parameter."""
    if isinstance(k_type, int):
        try:
            return type_number_to_pykdb_k_type[k_type]
        except KeyError:
            raise ValueError(f'Numeric k_type {k_type} does not exist')
    elif k_type is None:
        return None
    try:
        if issubclass(k_type, k.K):
            return k_type
    except TypeError:
        pass
    raise ValueError(f'k_type {k_type} unrecognized')


def adapt(x: Any, k_type: Optional[Union[Type[k.K], int]] = None) -> Type[k.K]:
    """Converts a Python object into a k object.

    :param x: A Python object which is to be converted into a k object
    :param k_type: The type number or pykdb.k.K class matching the desired
        type of the pykdb.k.K object returned from this call, defaults to
        None, which results in the type being inferred from the data
    """
    if x is None:
        adapter = adapt_none
    elif isinstance(x, k.K):
        adapter = adapt_k
    elif isinstance(x, (int, *np_int_types)):
        if isinstance(x, (bool, np.bool_)) and k_type is None:
            k_type = k.BooleanAtom
        adapter = adapt_int
    elif isinstance(x, (float, *np_float_types)):
        adapter = adapt_float
    elif isinstance(x, str):
        adapter = adapt_str
    elif isinstance(x, bytes):
        adapter = adapt_bytes
    elif isinstance(x, UUID):
        adapter = adapt_UUID
    elif isinstance(x, list):
        adapter = adapt_list
    elif isinstance(x, tuple):
        adapter = adapt_tuple
    elif isinstance(x, dict):
        adapter = adapt_dict
    elif isinstance(x, np.ndarray):
        adapter = adapt_ndarray
        if k_type is None:
            if x.dtype.char == 'U':
                k_type = k.SymbolVector
            elif x.dtype == object and isinstance(x[0], str):
                k_type = k.SymbolVector
                x = x.astype('U')
            elif x.dtype.char == 'S':
                k_type = k.CharVector
            elif x.dtype == np.float64:
                k_type = k.FloatVector
            elif x.dtype == np.float32:
                k_type = k.RealVector
            elif x.dtype == np.int64:
                k_type = k.LongVector
            elif x.dtype == np.int32:
                k_type = k.IntVector
            elif x.dtype == np.int16:
                k_type = k.ShortVector
            elif x.dtype == 'timedelta64[ms]':
                k_type = k.TimeVector
            elif x.dtype == 'timedelta64[ns]':
                k_type = k.TimespanVector
            elif x.dtype == 'datetime64[ns]':
                k_type = k.TimestampVector
            elif x.dtype == 'timedelta64[s]':
                k_type = k.SecondVector
            elif x.dtype == 'datetime64[M]':
                k_type = k.MonthVector
            elif x.dtype == 'datetime64[D]':
                k_type = k.DateVector
            elif x.dtype == 'timedelta64[m]':
                k_type = k.MinuteVector
            elif x.dtype == np.int8:
                k_type = k.ByteVector
            elif (x.dtype == object and isinstance(x[0], UUID)) or x.dtype == complex:  # noqa
                k_type = k.GUIDVector
            elif x.dtype == bool:
                k_type = k.BooleanVector
            elif x.dtype == object:
                k_type = k.List
            else:
                raise TypeError('k type cannot be inferred from Numpy '
                                f'dtype {x.dtype}')
    elif isinstance(x, pd.DataFrame):
        adapter = adapt_dataframe
    elif isinstance(x, pd.Series):
        adapter = adapt_series
    elif isinstance(x, pd.Index):
        adapter = adapt_index
    elif isinstance(x, pa.Table) or isinstance(x, pa.Array):
        adapter = adapt_arrow
    elif isinstance(x, datetime.datetime):
        adapter = adapt_datetime
    elif isinstance(x, datetime.timedelta):
        adapter = adapt_timedelta
    elif isinstance(x, np.datetime64):
        adapter = adapt_datetime64
    elif isinstance(x, np.timedelta64):
        adapter = adapt_timedelta64
    elif isinstance(x, slice):
        adapter = adapt_slice
    elif isinstance(x, range):
        adapter = adapt_range
    elif hasattr(x, 'fileno'):
        adapter = adapt_fileno
    else:
        adapter = default_adapter
    return adapter(x, resolve_ktype(k_type))


def default_adapter(x, k_type=None):
    raise TypeError(
        f"Cannot convert Python object {x!r} of type {type(x)} to K object")


def adapt_none(x, k_type=None):
    cdef kdb_api.K kx
    kx = kdb_api.ka(101)
    kx.g = 0
    return k.factory(<uintptr_t>kx)


def adapt_k(x, k_type=None):
    return k.factory(x._addr)


def adapt_int(x, k_type=None):
    cdef kdb_api.K kx
    if k_type is None or issubclass(k_type, k.LongAtom):
        kx = kdb_api.kj(x)
    elif issubclass(k_type, k.IntAtom):
        kx = kdb_api.ki(x)
    elif issubclass(k_type, k.ShortAtom):
        kx = kdb_api.kh(x)
    elif issubclass(k_type, k.ByteAtom):
        kx = kdb_api.kg(x)
    elif issubclass(k_type, k.BooleanAtom):
        kx = kdb_api.kb(x)
    else:
        raise TypeError(f'Cannot convert Python int to {k_type}')
    return k.factory(<uintptr_t>kx)


def adapt_float(x, k_type=None):
    cdef kdb_api.K kx
    if k_type is None or issubclass(k_type, k.FloatAtom):
        kx = kdb_api.kf(x)
    elif issubclass(k_type, k.RealAtom):
        kx = kdb_api.ke(x)
    else:
        raise TypeError(f'Cannot convert Python float {x} to {k_type}')
    return k.factory(<uintptr_t>kx)


def adapt_str(x, k_type=None):
    cdef kdb_api.K kx
    cdef bytes as_bytes = x.encode('utf-8')
    if k_type is None or issubclass(k_type, k.SymbolAtom):
        kx = kdb_api.ks(as_bytes)
    elif issubclass(k_type, k.CharAtom):
        kx = kdb_api.kc(ord(x) if x else 32)
    elif issubclass(k_type, k.CharVector):
        kx = kdb_api.kpn(as_bytes, len(as_bytes))
    else:
        raise TypeError(f'Cannot convert Python str {x} to {k_type}')
    return k.factory(<uintptr_t>kx)


def adapt_bytes(x, k_type=None):
    cdef kdb_api.K kx
    if (k_type is None and len(x) == 1) or (k_type is not None and issubclass(k_type, k.CharAtom)):  # noqa
        kx = kdb_api.kc(ord(x) if x else 32)
    elif k_type is None or issubclass(k_type, k.CharVector):
        kx = kdb_api.kpn(x, len(x))
    elif issubclass(k_type, k.SymbolAtom):
        kx = kdb_api.ks(x)
    else:
        raise TypeError(f'Cannot convert Python bytes {x} to {k_type}')
    return k.factory(<uintptr_t>kx)


def adapt_UUID(x, k_type=None):
    cdef kdb_api.K kx
    if k_type is not None and k_type != k.GUIDAtom:
        raise TypeError(f'Cannot convert uuid.UUID {x} to {k_type}')

    u = x.int
    u = (u & 0x0000000000000000FFFFFFFFFFFFFFFF) << 64 | (u & 0xFFFFFFFFFFFFFFFF0000000000000000) >> 64  # noqa
    u = (u & 0x00000000FFFFFFFF00000000FFFFFFFF) << 32 | (u & 0xFFFFFFFF00000000FFFFFFFF00000000) >> 32  # noqa
    u = (u & 0x0000FFFF0000FFFF0000FFFF0000FFFF) << 16 | (u & 0xFFFF0000FFFF0000FFFF0000FFFF0000) >> 16  # noqa
    u = (u & 0x00FF00FF00FF00FF00FF00FF00FF00FF) << 8  | (u & 0xFF00FF00FF00FF00FF00FF00FF00FF00) >> 8   # noqa
    cdef uint64_t upper_bits = (u & (-1 ^ 0xFFFFFFFFFFFFFFFF)) >> 64
    cdef uint64_t lower_bits = u & 0xFFFFFFFFFFFFFFFF
    cdef uint64_t data[2]
    data[0] = lower_bits
    data[1] = upper_bits
    cdef kdb_api.U guid
    guid.g = <char*>data
    return k.factory(<uintptr_t>kdb_api.ku(guid))


def adapt_list(x, k_type=None):
    if k_type is not None and not issubclass(k_type, k.Vector):
        raise TypeError(f'Cannot convert Python list {x} to {k_type}')
    cdef kdb_api.K kx = kdb_api.ktn(0, len(x))
    for i, item in enumerate(x):
        # No good way to specify the k_type for nested types
        (<kdb_api.K*>kx.G0)[i] = kdb_api.r1(_k(adapt(item)))
    wrapped = k.factory(<uintptr_t>kx)
    return wrapped


def adapt_tuple(x, k_type=None):
    if k_type is not None and not issubclass(k_type, k.Vector):
        raise TypeError(f'Cannot convert Python tuple {x} to {k_type}')
    return adapt_list(x)


def adapt_dict(x, k_type=None):
    if k_type is not None and k_type != k.Dictionary:
        raise TypeError(f'Cannot convert Python dict {x} to {k_type}')
    cdef kdb_api.K kx
    if all(isinstance(key, (str, k.SymbolAtom)) for key in x.keys()):
        k_keys = adapt(np.array([str(key) for key in x.keys()]))
    else:
        k_keys = adapt(list(x.keys()))
    k_values = adapt(list(x.values()))
    kx = kdb_api.xD(kdb_api.r1(_k(k_keys)), kdb_api.r1(_k(k_values)))
    return k.factory(<uintptr_t>kx)


# keys are supported ndarray k types, values are # of bytes per element
supported_nd_temporal_types = {
    k.TimestampVector: 8,
    k.MonthVector: 4,
    k.DateVector: 4,
    k.TimespanVector: 8,
    k.MinuteVector: 4,
    k.SecondVector: 4,
    k.TimeVector: 4,
}


supported_nd_nontemporal_types = {
    k.List: 8,
    k.BooleanVector: 1,
    k.GUIDVector: 16,
    k.ByteVector: 1,
    k.ShortVector: 2,
    k.IntVector: 4,
    k.LongVector: 8,
    k.RealVector: 4,
    k.FloatVector: 8,
    k.CharVector: 1,
    k.SymbolVector: 8,
}


supported_ndarray_k_types = dict(
    list(supported_nd_temporal_types.items())
    + list(supported_nd_nontemporal_types.items()))


def _listify(x):
    '''Convert all arrays except the lowest level into lists.'''
    if len(x.shape) > 1:
        return [_listify(y) for y in list(x)]
    return x


def adapt_ndarray(x, k_type=None):
    if k_type not in supported_ndarray_k_types:
        raise TypeError(f'Cannot convert numpy.ndarray {x} to {k_type}')

    # kdb doesn't support n-dimensional vectors, so we have to treat them as
    # lists to preserve the shape
    if len(x.shape) > 1:
        return adapt_list(_listify(x))

    if issubclass(k_type, k.CharVector) and x.dtype.itemsize > 1:
        return adapt_list(list(x))

    cdef kdb_api.J n = x.size
    cdef kdb_api.K kx = kdb_api.ktn(k_type.t, n)
    cdef bytes as_bytes
    cdef uintptr_t data

    if k_type not in supported_nd_temporal_types:
        if hasattr(x.data, 'contiguous') and not x.data.contiguous:
            raise ValueError('Cannot convert non-contiguous '
                             'ndarray to k vector')

    if issubclass(k_type, k.GUIDVector) and x.dtype == object:
        for i in range(n):
            (<np.complex128_t*>kx.G0)[i] = UUID_to_complex(x[i])
    elif issubclass(k_type, k.SymbolVector):
        for i in range(n):
            as_bytes = bytes(x[i], 'utf-8')
            (<char**>kx.G0)[i] = kdb_api.sn(as_bytes, len(as_bytes))
    elif k_type in supported_nd_temporal_types:
        if (issubclass(k_type, k.TimestampVector)
                or issubclass(k_type, k.TimespanVector)):
            offset = TIMESTAMP_OFFSET if issubclass(k_type, k.TimestampVector) else 0 # noqa
            for i in range(n):
                (<long*>kx.G0)[i] = x[i] - offset
        else:
            if issubclass(k_type, k.MonthVector):
                offset = MONTH_OFFSET
            elif issubclass(k_type, k.DateVector):
                offset = DATE_OFFSET
            else:
                offset = 0
            for i in range(n):
                (<int*>kx.G0)[i] = x.astype(int)[i] - offset
    else:
        if (k_type in supported_nd_nontemporal_types
                and not issubclass(k_type, k.List)):
            data = x.__array_interface__['data'][0]
            memcpy(
                <void*>kx.G0,
                <void*>data,
                n * supported_ndarray_k_types[k_type]
            )
        else:
            try:
                return adapt_list(x.tolist())
            except KdbError:
                if issubclass(k_type, k.List):
                    raise TypeError('Cannot convert non adaptable'
                                    f' numpy array to {k_type}')
    return k.factory(<uintptr_t>kx)


def adapt_dataframe(x, k_type=None):
    cdef kdb_api.K kx

    if k_type is None:
        if pd.Index(np.arange(0, len(x))).equals(x.index):
            k_type = k.Table
        else:
            k_type = k.KeyedTable

    if issubclass(k_type, k.Table):
        kx = kdb_api.xT(kdb_api.r1(_k(adapt_dict(
            {k: v.to_numpy() for k, v in x.iteritems()}))))
        if kx == NULL:
            raise PyKdbException('Failed to create table from k dictionary')
    elif issubclass(k_type, k.KeyedTable):
        k_keys = adapt_index(x.index)
        values = [x[c].values for c in x.columns]
        k_values = adapt_dataframe(
            df_from_arrays(x.columns, values, np.arange(len(x))))
        kx = kdb_api.xD(kdb_api.r1(_k(k_keys)), kdb_api.r1(_k(k_values)))
        if kx == NULL:
            raise PyKdbException('Failed to create k dictionary (keyed table)')
    else:
        raise TypeError(f'Cannot convert pandas.DataFrame {x} to {k_type}')
    return k.factory(<uintptr_t>kx)


def adapt_series(x, k_type=None):
    # Convert series to numpy, this is zero copy where possible by default
    # https://pandas.pydata.org/pandas-docs/version/0.24.0rc1/api/generated/pandas.Series.to_numpy.html
    # This conversion mechanism removes need for duplication of work to handle
    # individually typed pandas datatypes
    return adapt(x.to_numpy())


def adapt_index(x, k_type=None):
    if isinstance(x, pd.core.indexes.numeric.NumericIndex):
        # This handles DatetimeIndex and TimedeltaIndex in addition to all of
        # the regular int/float indexes
        return adapt(x.to_numpy())
    elif isinstance(x, pd.MultiIndex):
        d = {}
        for i in range(x.nlevels):
            d[x.levels[i].name] = x.levels[i][x.codes[i]].to_numpy()
	index_dict = adapt_dict(d)
	return k.factory(<uintptr_t>kdb_api.xT(_k(index_dict)))
    raise TypeError(f'Cannot convert {type(x)} {x} to k object')


def adapt_arrow(x, k_type=None):
    # Convert arrow array/table to pandas, this conversion is zero copy under
    # certain circumstances and is likely more efficient to convert largely
    # on python side where possible the below outlines limitations of this
    # https://arrow.apache.org/docs/python/pandas.html#memory-usage-and-zero-copy
    if isinstance(x, pa.ExtensionArray):
        raise TypeError(f'Cannot convert {type(x)} {x} to k object, '
                        'GUID array conversion not presently supported')
    return adapt(x.to_pandas())


def adapt_datetime(x, k_type=None):
    cdef kdb_api.K kx
    epoch = datetime.datetime(2000, 1, 1)
    if k_type is None or issubclass(k_type, k.TimestampAtom):
        d = x - epoch
        t = 1000 * (d.microseconds + 1000000 * (d.seconds + d.days * 86400))
        kx = kdb_api.ktj(-12, t)
    elif issubclass(k_type, k.MonthAtom):
        kx = kdb_api.ki(12 * (x.year - epoch.year) + x.month - epoch.month)
        kx.t = -13
    elif issubclass(k_type, k.DateAtom):
        kx = kdb_api.kd((x - epoch).days)
    elif issubclass(k_type, k.DatetimeAtom):
        raise NotImplementedError
    else:
        raise TypeError(f'Cannot convert datetime.datetime {x} to {k_type}')
    return k.factory(<uintptr_t>kx)


def adapt_timedelta(x, k_type=None):
    cdef kdb_api.K kx
    if k_type is None or issubclass(k_type, k.TimespanAtom):
        t = 1000 * (x.microseconds + 1000000 * (x.seconds + x.days * 86400))
        kx = kdb_api.ktj(-16, t)
    elif issubclass(k_type, k.MinuteAtom):
        kx = kdb_api.ki(x.total_seconds() // 60)
        kx.t = -17
    elif issubclass(k_type, k.SecondAtom):
        kx = kdb_api.ki(x.total_seconds())
        kx.t = -18
    elif issubclass(k_type, k.TimeAtom):
        kx = kdb_api.kt(
            x.days * 86400000 + x.seconds * 1000 + x.microseconds // 1000)
    else:
        raise TypeError(f'Cannot convert datetime.timedelta {x} to {k_type}')
    return k.factory(<uintptr_t>kx)


def adapt_datetime64(x, k_type=None):
    cdef kdb_api.K kx
    if k_type is None or issubclass(k_type, k.TimestampAtom):
        kx = kdb_api.ktj(
            -12,
            x.astype(np.dtype('datetime64[ns]')).astype(int) - TIMESTAMP_OFFSET
        )
    elif issubclass(k_type, k.MonthAtom):
        kx = kdb_api.ki(
            x.astype(np.dtype('datetime64[M]')).astype(int) - MONTH_OFFSET)
        kx.t = -13
    elif issubclass(k_type, k.DateAtom):
        kx = kdb_api.kd(
            x.astype(np.dtype('datetime64[D]')).astype(int) - DATE_OFFSET)
    elif issubclass(k_type, k.DatetimeAtom):
        raise NotImplementedError
    else:
        raise TypeError(f'Cannot convert numpy.datetime64 {x} to {k_type}')
    return k.factory(<uintptr_t>kx)


def adapt_timedelta64(x, k_type=None):
    cdef kdb_api.K kx
    if k_type is None or issubclass(k_type, k.TimespanAtom):
        kx = kdb_api.ktj(-16,
                         x.astype(np.dtype('timedelta64[ns]')).astype(int))
    elif issubclass(k_type, k.MinuteAtom):
        kx = kdb_api.ki(x.astype(np.dtype('timedelta64[m]')).astype(int))
        kx.t = -17
    elif issubclass(k_type, k.SecondAtom):
        kx = kdb_api.ki(x.astype(np.dtype('timedelta64[s]')).astype(int))
        kx.t = -18
    elif issubclass(k_type, k.TimeAtom):
        kx = kdb_api.kt(x.astype(np.dtype('timedelta64[ms]')).astype(int))
    else:
        raise TypeError(f'Cannot convert numpy.timedelta64 {x} to {k_type}')
    return k.factory(<uintptr_t>kx)


def adapt_slice(x, k_type=None):
    if x.stop is None:
        raise ValueError('Cannot convert endless slice to vector')
    return adapt_ndarray(np.asarray(slice_to_range(x, x.stop)),
                         k_type=k.LongVector)


def adapt_range(x, k_type=None):
    return adapt_ndarray(np.asarray(x), k_type=k.LongVector)


def adapt_fileno(x, k_type=None):
    if callable(x.fileno):
        x = x.fileno()
    else:
        x = x.fileno
    return adapt_int(x, k_type=k.IntAtom)


adapted_callables = {}

