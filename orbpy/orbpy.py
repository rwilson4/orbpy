import math
import sys

def rotate(v, axis, angle):
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
    defined.

    If the inclination is zero and the eccentricity is zero, use true_longitude.
    If the inclination is zero and the eccentricity is not zero, use true_anomaly (periapsis still defined).
    If the inclination is not zero and the eccentricity is zero, use arg_lat.
    If the inclination is not zero and the eccentricity is not zero, use true_anomaly.

    """

    ZERO_ECC = 1e-12
    ZERO_INC = 1e-12

    TRUE_ANOMALY = 0
    MEAN_ANOMALY = 1
    ECC_ANOMALY = 2

    def __init__(self, sma, ecc, inc, raan=None, argp=None,
                 true_anomaly=None, mean_anomaly=None,
                 eccentric_anomaly=None, arg_lat=None,
                 true_long=None, GM=3.986004418e14):
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
        return self._sma

    def get_ecc(self):
        return self._ecc

    def get_inc(self):
        return math.degrees(self._inc)

    def get_raan(self):
        return math.degrees(self._raan)

    def get_argp(self):
        return math.degrees(self._argp)

    def get_true_anomaly(self):
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
        # From Rene Schwarz
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
    def __init__(self, GM=3.986004418e14):
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

    def range(self, orbit, time):
        prop = KeplerianPropagator()
        prop.set_state(orbit._elements, orbit._epoch)
        elements = prop.propagate(time)
        r, v = elements.get_orbital_vectors()
        rx, ry, rz = r[0], r[1], r[2]
        
        # Let w be the angular velocity of the Earth.
        # Let t be the amount of time between the epoch (e.g. J2000) and time.
        # Let phi = w*t. Need to rotate about z by phi.
        x = rotate(self._x, 'z', self.W_EARTH * (time - self.J2000))
        ds2 = 0.
        for i in range(2):
            ds = r[i] - x[i]
            ds2 += ds * ds
        return math.sqrt(ds2)

if __name__ == '__main__':
    ce = ClassicalElements(sma=20000., ecc=0.1, inc=15., raan=20., argp=30., true_anomaly=0.)
    orb = Orbit(ce, 0.)
    kp = KeplerianPropagator()
    kp.set_state(ce, 0.)
    new_ce = kp.propagate(600.) # 10 minutes
    print new_ce.get_true_anomaly()
    la = GroundStation(lat=34, lon=118, alt=0.)
    print la.range(orb, 600.)
