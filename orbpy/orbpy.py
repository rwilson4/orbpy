import math
import sys

MU_EARTH = 398600.4418

def rotate(v, axis, angle):
    """Rotate a vector about an axis through an angle

    Parameters
    ----------
    v : array with exactly 3 numeric entries
        The vector to be rotated.
    axis : one of 'x', 'z'
        The axis about which to rotate. Rotations about y are not
        supported since they are not needed!
    angle : float
        Angle of rotation, in radians.

    Returns
    -------
    u : array with exactly 3 numeric entries
        The rotated vector.

    """
    u = [0, 0, 0]
    if axis == 'x':
        u[0] = v[0]
        u[1] = v[1] * math.cos(angle) - v[2] * math.sin(angle)
        u[2] = v[1] * math.sin(angle) + v[2] * math.cos(angle)
    elif axis == 'z':
        u[0] = v[0] * math.cos(angle) - v[1] * math.sin(angle)
        u[1] = v[0] * math.sin(angle) + v[1] * math.cos(angle)
        u[2] = v[2]
    else:
        raise ValueError('Not supported.')

    return u

class ClassicalElements:
    """Orbital Elements in Classical Form.

    Parameters
    ----------
    sma : float
        Semimajor axis, in kilometers
    ecc : float
        Eccentricity
    inc : float
        Inclination, in degrees
    raan : float
        Right ascension of the ascending node, in degrees
    argp : float
        Argument of periapsis, in degrees
    true_anomaly : float
        True anomaly, in degrees (see Notes)
    mean_anomaly : float
        Mean anomaly, in degrees (see Notes)
    ecc_anomaly : float
        Eccentric anomaly, in degrees (see Notes)
    arg_lat : float
        Argument of latitude, in degrees (see Notes)
    true_long : float
        True longitude, in degrees (see Notes)

    Usage
    -----
    ce = ClassicalElements(sma=3000, # km
                           ecc=0.003,
                           inc=15, # deg
                           raan=10, # deg
                           argp=20, # deg
                           true_anomaly=30 # deg
                          )

    Notes
    -----
    The argument of periapsis is defined relative to the ascending
    node. If the inclination is zero, this is not defined. In this
    case, we set the RAAN to be zero; this has the effect of defining
    the argument of periapsis relative to the same reference point as
    is used for the ascending node, (e.g. the point of Ares).

    The position within the orbit (true/mean/eccentric anomalies) are
    all defined relative to the argument of periapsis. In the event of
    a circular orbit (eccentricity equal to zero), this point is not
    defined. In that case, the position may be specified relative to
    the ascending node (arg_lat). If the ascending node is not
    defined, the position may be specified relative to the reference
    point (true_long).

    """

    ZERO_ECC = 1e-12
    ZERO_INC = 1e-12

    TRUE_ANOMALY = 0
    MEAN_ANOMALY = 1
    ECC_ANOMALY = 2

    def __init__(self, sma, ecc, inc, raan=None, argp=None,
                 true_anomaly=None, mean_anomaly=None,
                 eccentric_anomaly=None, arg_lat=None,
                 true_long=None, GM=MU_EARTH):
        self._sma = sma
        self._ecc = ecc
        self._inc = math.radians(inc)
        self._GM = GM

        if self._inc <= self.ZERO_INC:
            self._raan = 0
        elif raan is not None:
            self._raan = math.radians(raan)
        else:
            raise ValueError('RAAN must be provided')

        if self._ecc <= self.ZERO_ECC:
            self._argp = 0
        elif argp is not None:
            self._argp = math.radians(argp)
        else:
            raise ValueError('Arg. Periapsis must be provided')

        if true_anomaly is not None:
            self._anomaly_type = self.TRUE_ANOMALY
            self._true_anomaly = math.radians(true_anomaly)
        elif mean_anomaly is not None:
            self._anomaly_type = self.MEAN_ANOMALY
            self._mean_anomaly = math.radians(mean_anomaly)
        elif ecc_anomaly is not None:
            self._anomaly_type = self.ECC_ANOMALY
            self._ecc_anomaly = math.radians(ecc_anomaly)
        elif arg_lat is not None:
            self._anomaly_type = self.TRUE_ANOMALY
            self._true_anomaly = math.radians(arg_lat) - self._argp
        elif true_long is not None:
            self._anomaly_type = self.TRUE_ANOMALY
            self._true_anomaly = math.radians(true_long) - self.argp - self.raan
        else:
            raise ValueError('Position in orbit must be provided (e.g. true anomaly).')


    def get_sma(self):
        '''Get semimajor axis (in km)'''
        return self._sma


    def get_ecc(self):
        '''Get eccentricity'''
        return self._ecc


    def get_inc(self):
        '''Get inclination (in degrees)'''
        return math.degrees(self._inc)


    def get_raan(self):
        '''Get RAAN (in degrees)'''
        return math.degrees(self._raan)


    def get_argp(self):
        '''Get argument of periapsis (in degrees)'''
        return math.degrees(self._argp)


    def get_true_anomaly(self):
        '''Get true anomaly (in degrees)'''
        if self._anomaly_type == self.TRUE_ANOMALY:
            return math.degrees(self._true_anomaly)
        elif self._anomaly_type == self.ECC_ANOMALY:
            true_anomaly = self.ecc_to_true(self._ecc_anomaly, self.get_ecc())
            return math.degrees(true_anomaly)
        elif self._anomaly_type == self.MEAN_ANOMALY:
            ecc_anomaly = self.mean_to_ecc(self._mean_anomaly, self.get_ecc())
            true_anomaly = self.ecc_to_true(ecc_anomaly, self.get_ecc())
            return math.degrees(true_anomaly)


    def get_mean_anomaly(self):
        '''Get mean anomaly (in degrees)'''
        if self._anomaly_type == self.TRUE_ANOMALY:
            ecc_anomaly = self.true_to_ecc(self._true_anomaly, self.get_ecc())
            mean_anomaly = self.ecc_to_mean(ecc_anomaly, self.get_ecc())
            return math.degrees(mean_anomaly)
        elif self._anomaly_type == self.ECC_ANOMALY:
            mean_anomaly = self.ecc_to_mean(self._ecc_anomaly, self.get_ecc())
            return math.degrees(mean_anomaly)
        elif self._anomaly_type == self.MEAN_ANOMALY:
            return math.degrees(self._mean_anomaly)


    def get_ecc_anomaly(self):
        '''Get eccentric anomaly (in degrees)'''
        if self._anomaly_type == self.TRUE_ANOMALY:
            ecc_anomaly = self.true_to_ecc(self._true_anomaly, self.get_ecc())
            return math.degrees(ecc_anomaly)
        elif self._anomaly_type == self.ECC_ANOMALY:
            return math.degrees(self._ecc_anomaly)
        elif self._anomaly_type == self.MEAN_ANOMALY:
            ecc_anomaly = self.mean_to_ecc(self._mean_anomaly, self.get_ecc())
            return math.degrees(ecc_anomaly)


    def ecc_to_true(self, ecc_anomaly, eccentricity):
        '''Convert eccentric anomaly to true anomaly.'''
        return 2. * math.atan2(math.sqrt(1 + eccentricity) * math.sin(ecc_anomaly / 2.),
                               math.sqrt(1 - eccentricity) * math.cos(ecc_anomaly / 2.))


    def true_to_ecc(self, true_anomaly, eccentricity):
        '''Convert true anomaly to eccentric anomaly.'''
        return math.acos((eccentricity + math.cos(true_anomaly)) / (1. + eccentricity * math.cos(true_anomaly)))


    def mean_to_ecc(self, mean_anomaly, eccentricity):
        '''Convert mean anomaly to eccentric anomaly.'''
        # Given M, e, find E s.t. 0 = E - e sin(E) - M = f(E)
        # E_0 = M
        # E_{k+1} = E_k - f(E_k) / f'(E_k)
        E = mean_anomaly
        tol = 1e-6
        max_its = 100
        for i in range(max_its):
            f = E - eccentricity * math.sin(E) - mean_anomaly
            if abs(f) <= tol:
                break
            fp = 1 - eccentricity * math.cos(E)
            E -= f / fp
        else:
            raise ValueError('Did not converge')

        return E


    def ecc_to_mean(self, ecc_anomaly, eccentricity):
        '''Convert eccentric anomaly to mean anomaly.'''
        return ecc_anomaly - eccentricity * math.sin(ecc_anomaly)


    def get_orbital_vectors(self):
        """Convert classical elements to orbital vectors

        Notes
        -----
        We use the method of Rene Schwarz.
        https://www.rene-schwarz.com/web/Home
        https://downloads.rene-schwarz.com/download/M001-Keplerian_Orbit_Elements_to_Cartesian_State_Vectors.pdf

        """

        E = self.get_ecc_anomaly()
        tru = self.get_true_anomaly()
        e = self.get_ecc()
        a = self.get_sma()
        rc = a * (1. - e * math.cos(E))
        ot = [rc * math.cos(tru), rc * math.sin(tru), 0.]
        c = math.sqrt(self._GM * a) / rc
        otd = [c * -math.sin(E), c * math.sqrt(1. - e * e) * math.cos(E), 0]

        r1 = rotate(ot, 'z', -self._argp)
        r2 = rotate(r1, 'x', -self._inc)
        r  = rotate(r2, 'z', -self._raan)

        v1 = rotate(otd, 'z', -self._argp)
        v2 = rotate(v1,  'x', -self._inc)
        v  = rotate(v2,  'z', -self._raan)

        return r, v


class KeplerianPropagator:
    """A simple 2-body propagator.

    Usage
    -----
    from orbpy import KeplerianProgator, ClassicalElements, ISS
    prop = KeplerianPropagator()
    prop.set_state(state)

    """
    def __init__(self, GM=MU_EARTH):
        self._GM = GM


    def set_state(self, state, epoch):
        self._state = state
        self._epoch = epoch


    def propagate(self, t):
        delta_t = t - self._epoch
        mean_anomaly_at_epoch = self._state.get_mean_anomaly()
        sma = self._state.get_sma()
        n = math.sqrt(self._GM / ( sma * sma * sma ))
        mean_anomaly_at_time = mean_anomaly_at_epoch + n * delta_t

        return ClassicalElements(sma=sma,
                                 ecc=self._state.get_ecc(),
                                 inc=self._state.get_inc(),
                                 raan=self._state.get_raan(),
                                 argp=self._state.get_argp(),
                                 mean_anomaly=mean_anomaly_at_time)


class Orbit:
    def __init__(self, elements, epoch):
        self._elements = elements
        self._epoch = epoch


    def propagate(self, t, propagator='kepler'):
        if propagator != 'kepler':
            raise NotImplementedError('Only Keplerian Propagator supported currently.')

        prop = KeplerianPropagator()
        prop.set_state(self._elements, self._epoch)
        self._elements = prop.propagate(t)
        self._epoch = t


class GroundStation:
    """Ground station class

    Usage
    -----
    ISS = ClassicalElements()
    ISS_epoch = 0
    los_angeles = GroundStation(lat=34, lon=118)
    los_angeles.range(ISS, 10)

    """

    R_EARTH = 6378.14 # km
    J2000 = 0. # seconds
    W_EARTH = 2 * math.pi / 86400. # radians per second

    def __init__(self, lat, lon, alt=0.):
        self._lat = math.radians(lat)
        self._lon = math.radians(lon)
        self._alt = alt
        self._x = self.lla_to_xyz(self._lat, self._lon, self._alt)


    def lla_to_xyz(self, lat, lon, alt):
        x = [0, 0, 0]
        x[0] = (self.R_EARTH + alt) * math.cos(lat) * math.cos(lon)
        x[1] = (self.R_EARTH + alt) * math.cos(lat) * math.sin(lon)
        x[2] = (self.R_EARTH + alt) * math.sin(lat)

        # Rotate to account for Earth's spin
        # Let theta be the angle 0 lat, 0 lon made with respect to the
        # reference point at the epoch (e.g. J2000).
        theta = 0.
        return rotate(x, 'z', theta)


    def range_az_el(self, orbit, time):
        prop = KeplerianPropagator()
        prop.set_state(orbit._elements, orbit._epoch)
        elements = prop.propagate(time)
        r, v = elements.get_orbital_vectors()

        # Let w be the angular velocity of the Earth.
        # Let t be the amount of time between the epoch (e.g. J2000) and time.
        # Let phi = w*t. Need to rotate about z by phi.
        x = rotate(self._x, 'z', self.W_EARTH * (time - self.J2000))

        # Range is just \| r - x \|_2
        ds2 = 0.
        for i in range(2):
            ds = r[i] - x[i]
            ds2 += ds * ds
        rng = math.sqrt(ds2)

        # El is the angle between (r - x) and x
        # cos(El) = (r - x) dot x / (norm x * norm (r-x))
        num = 0.
        nx = 0.
        nrmx = 0.
        for i in range(2):
            num += x[i] * (r[i] - x[i])
            nx += x[i] * x[i]
            nrmx += (r[i] - x[i]) * (r[i] - x[i])
        el = math.degrees(math.acos(num / math.sqrt(nx * nrmx)))

        # Az is more complicated and not needed for estpy, so I'm skipping!
        az = 0
        return rng, az, el


if __name__ == '__main__':
    ce = ClassicalElements(sma=20000., ecc=0.1, inc=15., raan=20., argp=30., true_anomaly=0.)
    orb = Orbit(ce, 0.)
    kp = KeplerianPropagator()
    kp.set_state(ce, 0.)
    new_ce = kp.propagate(600.) # 10 minutes
    print new_ce.get_true_anomaly()
    la = GroundStation(lat=34, lon=118, alt=0.)
    for it in range(60):
        r, az, el = la.range_az_el(orb, 10. * it)
        print 10. * it, r, el
