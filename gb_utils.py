import numpy as np
from scipy.interpolate import interp1d
import math
from scipy import ndimage, signal


def fill_check(array):
    array2 = np.array(array)
    x = np.arange(len(array2))
    idx = np.nonzero(array2)
    interp = interp1d(x[idx], array2[idx])
    new_dat = interp(x)
    return new_dat


def getAngle(a, b, c):
    """Counterclockwise angle in degrees by turning from a to c around b
    Returns a float between 0.0 and 360.0"""
    ang = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    return ang + 360 if ang < 0 else ang


def lineAngle(p1x, p1y, p2x, p2y):  # Angle relative to vertical
    deltaY = p2y - p1y
    deltaX = p2x - p1x
    ang = math.degrees(math.atan2(deltaY, deltaX)) + 90
    if ang < 0 and ang > -90:
        ang = np.abs(ang)
    elif ang >= 180 and ang <= 270:
        ang = 360 - ang
    return ang


def filterG(input):
    dat = ndimage.median_filter(input, size=5)
    return dat


def lowpass(data_in, Hz):
    b, a = signal.butter(4, Hz / 100, btype='lowpass')
    output = signal.filtfilt(b, a, data_in)
    return output


def rep_count(input_sig):
    input_sig = np.array(input_sig)
    diff = np.diff(input_sig)
    diff = np.insert(diff, 0, 0)

    pos = np.copy(diff)
    pos[pos < 2] = 2
    peaks, inf = signal.find_peaks(pos)

    neg = np.copy(diff)
    neg[neg > -2] = -2
    troughs, infT = signal.find_peaks(-neg)

    if len(peaks) == len(troughs):
        reps = len(peaks)
    else:
        reps = len(troughs)  # Don't count half reps
    return reps


def distance(p1x, p1y, p2x, p2y):
    distance = math.sqrt(((p1x - p2x) ** 2) + ((p1y - p2y) ** 2))
    return (distance)

