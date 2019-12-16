# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 11:24:24 2015

@author: T.C. van Leth
"""

import phad as ha


def base_SI(invar):
    """
    convert units of var to base SI units
    """
    unit = invar.attrs['unit']
    quan = invar.attrs['quantity']

    if quan == 'pressure':
        newunit = 'Pa'
        if unit == 'kPa':
            def func(x): return x * 1000
        elif unit == 'hPa':
            def func(x): return x * 100
        elif unit == newunit or unit == 'pascal':
            return invar
        else:
            raise UnitError('%s input has unknown unit! "%s"' % (quan, unit))

    elif (quan == 'diameter' or quan == 'length' or quan == 'visibility' or
          quan == 'snow_height'):
        newunit = 'm'
        if unit == 'mm':
            def func(x): return x * 0.001
        elif unit == 'km':
            def func(x): return x * 1000
        elif unit == newunit or unit == 'meter':
            return invar
        else:
            raise UnitError('%s input has unknown unit! "%s"' % (quan, unit))

    elif quan == 'area' or quan == 'cross_section':
        newunit = 'm'
        if unit == 'mm2':
            def func(x): return x * 1e-6
        elif unit == newunit:
            return invar
        else:
            raise UnitError('%s input has unknown unit! "%s"' % (quan, unit))

    elif quan == 'frequency':
        newunit = 'Hz'
        if unit == 'GHz':
            def func(x): return x * 1e9
        elif unit == 'MHz':
            def func(x): return x * 1e6
        elif unit == newunit:
            return invar
        else:
            raise UnitError('%s input has unknown unit! "%s"' % (quan, unit))

    elif quan == 'temperature':
        newunit = 'K'
        if unit == 'celsius' or unit == 'deg_celsius' or unit == 'deg_C':
            def func(x): return x + 273.15 # TODO: fix for step!
        elif unit == 'kelvin' or unit == newunit:
            return invar
        else:
            raise UnitError('%s input has unknown unit! "%s"' % (quan, unit))

    elif (quan == 'velocity' or quan == 'rain_intensity' or
          quan == 'wind_speed'):
        newunit = 'm/s'
        if unit == 'mm/hr':
            def func(x): return x * (0.001/3600)
        elif unit == 'km/hr':
            def func(x): return x * (1000/3600)
        elif unit == 'mm/min':
            def func(x): return x * (0.001/60)
        elif unit == newunit:
            return invar
        else:
            raise UnitError('%s input has unknown unit! "%s"' % (quan, unit))

    elif quan == 'current':
        newunit = 'A'
        if unit == 'mA':
            def func(x): return x * 0.001
        elif unit == 'kA':
            def func(x): return x * 1000
        elif unit == newunit or unit == 'Ampere':
            return invar
        else:
            raise UnitError('%s input has unknown unit! "%s"' % (quan, unit))

    elif (quan == 'voltage' or quan == 'detector_voltage' or
          quan == 'suppply_voltage' or quan == 'ground_voltage'):
        newunit = 'V'
        if unit == 'mV':
            def func(x): return x * 0.001
        elif unit == 'kV':
            def func(x): return x * 1000
        elif unit == newunit or unit == 'volt':
            return invar
        else:
            raise UnitError('%s input has unknown unit! "%s"' % (quan, unit))

    elif quan == 'relative_humidity':
        newunit = 'fraction'
        if unit == 'percent':
            def func(x): return x * 0.01
        elif unit == newunit:
            return invar
        else:
            raise UnitError('%s input has unknown unit! "%s"' % (quan, unit))
    else:
        return invar

    outvar = func(invar)
    outvar.attrs['unit'] = newunit
    if hasattr(invar, '_start'):
        outvar._start = func(invar._start)
        outvar._stop = func(invar._stop)
    if hasattr(invar, '_step') and invar._step is not None:
        outvar._step = func(invar._step)
    return outvar


class UnitError(ha.common.Error):
    pass
