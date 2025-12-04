import math


def wgs84ToNED(lat, lon, h, lat0=24.8976763, lon0=160.123456, h0=0):
    # function[xEast, yNorth, zUp] = geodetic_to_enu(lat, lon, h, lat0, lon0, h0)
    a = 6378137
    b = 6356752.3142
    f = (a - b) / a
    e_sq = f * (2 - f)

    lamb = math.radians(lat)  # 角度换成弧度
    phi = math.radians(lon)
    s = math.sin(lamb)
    N = a / math.sqrt(1 - e_sq * s * s)

    sin_lambda = math.sin(lamb)
    cos_lambda = math.cos(lamb)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)

    x = (h + N) * cos_lambda * cos_phi
    y = (h + N) * cos_lambda * sin_phi
    z = (h + (1 - e_sq) * N) * sin_lambda

    # 原点坐标转换
    lamb0 = math.radians(lat0)
    phi0 = math.radians(lon0)
    s0 = math.sin(lamb0)
    N0 = a / math.sqrt(1 - e_sq * s0 * s0)

    sin_lambda0 = math.sin(lamb0)
    cos_lambda0 = math.cos(lamb0)
    sin_phi0 = math.sin(phi0)
    cos_phi0 = math.cos(phi0)

    x0 = (h0 + N0) * cos_lambda0 * cos_phi0
    y0 = (h0 + N0) * cos_lambda0 * sin_phi0
    z0 = (h0 + (1 - e_sq) * N0) * sin_lambda0

    xd = x - x0
    yd = y - y0
    zd = z - z0

    t = -cos_phi0 * xd - sin_phi0 * yd

    xEast = -sin_phi0 * xd + cos_phi0 * yd
    yNorth = t * sin_lambda0 + cos_lambda0 * zd
    zUp = cos_lambda0 * cos_phi0 * xd + cos_lambda0 * sin_phi0 * yd + sin_lambda0 * zd

    # return yNorth, xEast, -zUp
    return xEast, yNorth, zUp


def enu_to_ecef(xEast, yNorth, zUp, lat0=24.8976763, lon0=160.123456, h0=0):
    a = 6378137
    b = 6356752.3142
    f = (a - b) / a
    e_sq = f * (2 - f)
    # pi = 3.14159265359
    pi = math.pi

    lamb = math.radians(lat0)
    phi = math.radians(lon0)
    s = math.sin(lamb)
    N = a / math.sqrt(1 - e_sq * s * s)

    sin_lambda = math.sin(lamb)
    cos_lambda = math.cos(lamb)
    sin_phi = math.sin(phi)
    cos_phi = math.cos(phi)

    x0 = (h0 + N) * cos_lambda * cos_phi
    y0 = (h0 + N) * cos_lambda * sin_phi
    z0 = (h0 + (1 - e_sq) * N) * sin_lambda

    t = cos_lambda * zUp - sin_lambda * yNorth

    zd = sin_lambda * zUp + cos_lambda * yNorth
    xd = cos_phi * t - sin_phi * xEast
    yd = sin_phi * t + cos_phi * xEast

    x = xd + x0
    y = yd + y0
    z = zd + z0
    return x, y, z


def ecef_to_geodetic(x, y, z):
    # function[lat0, lon0, h0] = ecef_to_geodetic(x, y, z)
    # Convert from ECEF cartesian coordinates to
    # latitude, longitude and height.  WGS-84
    a = 6378137
    b = 6356752.3142
    f = (a - b) / a
    e_sq = f * (2 - f)
    # pi = 3.14159265359
    pi = math.pi

    x2 = x ** 2
    y2 = y ** 2
    z2 = z ** 2

    a = 6378137.0000  # earth radius in meters
    b = 6356752.3142  # earth semiminor in meters
    e = math.sqrt(1 - (b / a) ** 2)
    b2 = b * b
    e2 = e ** 2
    ep = e * (a / b)
    r = math.sqrt(x2 + y2)
    r2 = r * r
    E2 = a ** 2 - b ** 2
    F = 54 * b2 * z2
    G = r2 + (1 - e2) * z2 - e2 * E2
    c = (e2 * e2 * F * r2) / (G * G * G)
    s = (1 + c + math.sqrt(c * c + 2 * c)) ** (1 / 3)
    P = F / (3 * (s + 1 / s + 1) ** 2 * G * G)
    Q = math.sqrt(1 + 2 * e2 * e2 * P)
    ro = -(P * e2 * r) / (1 + Q) + math.sqrt(
        (a * a / 2) * (1 + 1 / Q) - (P * (1 - e2) * z2) / (Q * (1 + Q)) - P * r2 / 2)
    tmp = (r - e2 * ro) ** 2
    U = math.sqrt(tmp + z2)
    V = math.sqrt(tmp + (1 - e2) * z2)
    zo = (b2 * z) / (a * V)

    height = U * (1 - b2 / (a * V))

    lat = math.atan((z + ep * ep * zo) / r)

    temp = math.atan(y / x)
    if x >= 0:
        long = temp
    else:
        if (x < 0) & (y >= 0):
            long = pi + temp
        else:
            long = temp - pi

    lat0 = lat / (pi / 180)
    lon0 = long / (pi / 180)
    h0 = height
    return lat0, lon0, h0


def ned_to_wgs84(vec, lat0=24.8976763, lon0=160.123456, h0=0):
    x, y, z = enu_to_ecef(vec[1], vec[0], -vec[2], lat0, lon0, h0)
    lat, lon, h = ecef_to_geodetic(x, y, z)
    return lon, lat, h
