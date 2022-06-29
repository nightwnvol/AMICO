import numpy as np
import scipy
from amico.util import ERROR

# Limits the required precision in gpd sum
REQUIRED_PRECISION = 1e-7

# Gyromagnetic ratio
# GAMMA = 2.675153151e8 # https://physics.nist.gov/cgi-bin/cuu/Results?search_for=proton+gyromagnetic+ratio
# GAMMA = 2.6751525e8 # this is used in camino
GAMMA = 2.675987e8 # this is used in NODDI

def _gpd_sum(am, big_delta, small_delta, diff, radius, n):
    sum = 0.0
    for _am in am:
        dam = diff * _am * _am
        e11 = -dam * small_delta
        e2 = -dam * big_delta
        dif = big_delta - small_delta
        e3 = -dam * dif
        plus = big_delta + small_delta
        e4 = -dam * plus
        nom = 2 * dam * small_delta - 2 + (2 * np.exp(e11)) + (2 * np.exp(e2)) - np.exp(e3) - np.exp(e4)
        denom = dam * dam * _am * _am * (radius * radius * _am * _am - n)
        term = nom / denom
        sum += term
        if term < REQUIRED_PRECISION * sum:
            break
    return big_delta, small_delta, diff, radius, sum

# SPHERE
class SphereGPD():
    AM_SPHERE = np.array([
    2.081575978, 5.940369990, 9.205840145,
    12.40444502, 15.57923641, 18.74264558, 21.89969648,
    25.05282528, 28.20336100, 31.35209173, 34.49951492,
    37.64596032, 40.79165523, 43.93676147, 47.08139741,
    50.22565165, 53.36959180, 56.51327045, 59.65672900,
    62.80000055, 65.94311190, 69.08608495, 72.22893775,
    75.37168540, 78.51434055, 81.65691380, 84.79941440,
    87.94185005, 91.08422750, 94.22655255, 97.36883035,
    100.5110653, 103.6532613, 106.7954217, 109.9375497,
    113.0796480, 116.2217188, 119.3637645, 122.5057870,
    125.6477880, 128.7897690, 131.9317315, 135.0736768,
    138.2156061, 141.3575204, 144.4994207, 147.6413080,
    150.7831829, 153.9250463, 157.0668989, 160.2087413,
    163.3505741, 166.4923978, 169.6342129, 172.7760200,
    175.9178194, 179.0596116, 182.2013968, 185.3431756,
    188.4849481, 191.6267147, 194.7684757, 197.9102314,
    201.0519820, 204.1937277, 207.3354688, 210.4772054,
    213.6189378, 216.7606662, 219.9023907, 223.0441114,
    226.1858287, 229.3275425, 232.4692530, 235.6109603,
    238.7526647, 241.8943662, 245.0360648, 248.1777608,
    251.3194542, 254.4611451, 257.6028336, 260.7445198,
    263.8862038, 267.0278856, 270.1695654, 273.3112431,
    276.4529189, 279.5945929, 282.7362650, 285.8779354,
    289.0196041, 292.1612712, 295.3029367, 298.4446006,
    301.5862631, 304.7279241, 307.8695837, 311.0112420,
    314.1528990
    ])

    last_big_delta = 0.0
    last_small_delta = 0.0
    last_diff = 0.0
    last_radius = 0.0
    last_sum = 0.0

    def __init__(self, scheme):
        self.scheme = scheme

    def get_signal(self, diff, radius):
        diff *= 1e-6
        am = self.AM_SPHERE / radius
        signal = np.zeros(len(self.scheme.raw))
        for i, raw in enumerate(self.scheme.raw):
            g_dir = raw[0:3]
            g = raw[3]
            big_delta = raw[4]
            small_delta = raw[5]

            if np.all(g_dir == 0):
                signal[i] = 1
            else:
                g_mods = g_dir * g
                g_mod = np.sqrt(np.dot(g_mods, g_mods))
                # if big_delta == self.last_big_delta and small_delta == self.last_small_delta and self.diff == self.last_diff and self.radius == self.last_radius:
                #     sum = self.last_sum
                # else:
                #     self.last_big_delta, self.last_small_delta, self.last_diff, self.last_radius, self.last_sum = _gpd_sum(self.am, big_delta, small_delta, self.diff, self.radius, 2)
                #     sum = self.last_sum
                if big_delta != self.last_big_delta or small_delta != self.last_small_delta or diff != self.last_diff or radius != self.last_radius:
                    self.last_big_delta, self.last_small_delta, self.last_diff, self.last_radius, self.last_sum = _gpd_sum(am, big_delta, small_delta, diff, radius, 2)
                signal[i] = np.exp(-2 * GAMMA * GAMMA * g_mod * g_mod * self.last_sum)
        return signal

# ZEPPELIN
class Zeppelin():
    def __init__(self, scheme):
        self.scheme = scheme

        self.g_dir = self.scheme.raw[:, :3]
        self.b = self.scheme.b

    def get_signal(self, diff_par, diff_perp, theta, phi):
        n = np.array([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)])
        gn = np.dot(self.g_dir, n)
        signal = np.exp(-self.b * ((diff_par - diff_perp) * gn * gn + diff_perp))
        return signal

# STICK
class Stick():
    def __init__(self, scheme):
        self.scheme = scheme

        self.g_dir = self.scheme.raw[:, :3]
        self.b = self.scheme.b

    def get_signal(self, diff, theta, phi):
        n = np.array([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)])
        gn = np.dot(self.g_dir, n)
        signal = np.exp(-self.b * diff * gn * gn)
        return signal

# ASTROSTICKS
class Astrosticks():
    def __init__(self, scheme):
        self.scheme = scheme

    def get_signal(self, diff):
        signal = np.zeros(len(self.scheme.raw))
        for i, raw in enumerate(self.scheme.raw):
            g_dir = raw[0:3]
            g = raw[3]
            b = self.scheme.b[i]
            
            if np.all(g_dir == 0):
                signal[i] = 1
            else:
                l_perp = 0
                l_par = -b / (g * g) * diff
                signal[i] = np.sqrt(np.pi) * 1 / (2 * g * np.sqrt(l_perp - l_par)) * np.exp(g * g * l_perp) * scipy.special.erf(g * np.sqrt(l_perp - l_par))
        return signal

# BALL
class Ball():
    def __init__(self, scheme):
        self.scheme = scheme
        
        self.b = self.scheme.b

    def get_signal(self, diff):
        signal = np.exp(-self.b * diff)
        return signal

# CYLINDER
class CylinderGPD():
    AM_CYLINDER = np.array([
    1.84118307861360, 5.33144196877749, 
    8.53631578218074, 11.7060038949077, 14.8635881488839,
    18.0155278304879, 21.1643671187891, 24.3113254834588,
    27.4570501848623, 30.6019229722078, 33.7461812269726,
    36.8899866873805, 40.0334439409610, 43.1766274212415,
    46.3195966792621, 49.4623908440429, 52.6050411092602,
    55.7475709551533, 58.8900018651876, 62.0323477967829,
    65.1746202084584, 68.3168306640438, 71.4589869258787,
    74.6010956133729, 77.7431620631416, 80.8851921057280,
    84.0271895462953, 87.1691575709855, 90.3110993488875,
    93.4530179063458, 96.5949155953313, 99.7367932203820,
    102.878653768715, 106.020498619541, 109.162329055405,
    112.304145672561, 115.445950418834, 118.587744574512,
    121.729527118091, 124.871300497614, 128.013065217171,
    131.154821965250, 134.296570328107, 137.438311926144,
    140.580047659913, 143.721775748727, 146.863498476739,
    150.005215971725, 153.146928691331, 156.288635801966,
    159.430338769213, 162.572038308643, 165.713732347338,
    168.855423073845, 171.997111729391, 175.138794734935,
    178.280475036977, 181.422152668422, 184.563828222242,
    187.705499575101
    ])

    last_big_delta = 0.0
    last_small_delta = 0.0
    last_diff = 0.0
    last_radius = 0.0
    last_sum = 0.0

    def __init__(self, scheme):
        self.scheme = scheme

    def get_signal(self, diff, theta, phi, radius):
        diff *= 1e-6
        am = self.AM_CYLINDER / radius
        n = np.array([np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)])
        n_mod = np.sqrt(np.sum(n * n))
        signal = np.zeros(len(self.scheme.raw))
        for i, raw in enumerate(self.scheme.raw):
            g_dir = raw[0:3]
            g = raw[3]
            big_delta = raw[4]
            small_delta = raw[5]

            if np.all(g_dir == 0):
                signal[i] = 1
            else:
                g_mods = g_dir * g
                g_mod = np.sqrt(np.dot(g_mods, g_mods))
                gn = np.dot(g_mods, n)
                unit_gn = 0
                if g_mod == 0:
                    unit_gn = 0
                else:
                    unit_gn = gn / (g_mod * n_mod)
                omega = np.arccos(unit_gn)
                # if big_delta == self.last_big_delta and small_delta == self.last_small_delta and self.diff == self.last_diff and self.radius == self.last_radius:
                #     sum = self.last_sum
                # else:
                #     self.last_big_delta, self.last_small_delta, self.last_diff, self.last_radius, self.last_sum = _gpd_sum(am, big_delta, small_delta, self.diff, self.radius, 1)
                #     sum = self.last_sum
                if big_delta != self.last_big_delta or small_delta != self.last_small_delta or diff != self.last_diff or radius != self.last_radius:
                    self.last_big_delta, self.last_small_delta, self.last_diff, self.last_radius, self.last_sum = _gpd_sum(am, big_delta, small_delta, diff, radius, 1)
                sr_perp = np.exp(-2 * GAMMA * GAMMA * g_mod * g_mod * np.sin(omega) * np.sin(omega) * self.last_sum)
                t = big_delta - small_delta / 3
                sr_par = np.exp(-t * (GAMMA * small_delta * g_mod * np.cos(omega) * (GAMMA * small_delta * g_mod * np.cos(omega))) * diff)
                signal[i]= sr_perp * sr_par
        return signal

# NODDI
class NODDISignal():
    def __init__(self, scheme):
        self.scheme = scheme

        self.protocol_hr = self._scheme2noddi(self.scheme)

    def get_ic_signal(self, diff_par, kappa):
        diff_par *= 1e-6
        return self._synth_meas_watson_SH_cyl_neuman_PGSE(
            np.array([diff_par, 0, kappa]),
            self.protocol_hr['grad_dirs'],
            np.squeeze(self.protocol_hr['gradient_strength']),
            np.squeeze(self.protocol_hr['delta']),
            np.squeeze(self.protocol_hr['smalldel']),
            np.array([0, 0, 1]))

    def get_ec_signal(self, diff_par, kappa, vol_ic):
        diff_par *= 1e-6
        diff_perp = diff_par * (1 - vol_ic)
        return self._synth_meas_watson_hindered_diffusion_PGSE(
            np.array([diff_par, diff_perp, kappa]),
            self.protocol_hr['grad_dirs'],
            np.squeeze(self.protocol_hr['gradient_strength']),
            np.squeeze(self.protocol_hr['delta']),
            np.squeeze(self.protocol_hr['smalldel']),
            np.array([0, 0, 1]))

    def get_iso_signal(self, diff_iso):
        diff_iso *= 1e-6
        return self._synth_meas_iso_GPD(diff_iso, self.protocol_hr)

    # intra-cellular signal
    def _synth_meas_watson_SH_cyl_neuman_PGSE(self, x, grad_dirs, G, delta, smalldel, fibredir):
        d=x[0]
        R=x[1]
        kappa=x[2]

        l_q = grad_dirs.shape[0]

        # Parallel component
        LePar = self._cyl_neuman_le_par_PGSE(d, G, delta, smalldel)

        # Perpendicular component
        LePerp = self._cyl_neuman_le_perp_PGSE(d, R, G)

        ePerp = np.exp(LePerp)

        # Compute the Legendre weighted signal
        Lpmp = LePerp - LePar
        lgi = self._legendre_gaussian_integral(Lpmp, 6)

        # Compute the spherical harmonic coefficients of the Watson's distribution
        coeff = self._watson_SH_coeff(kappa)
        coeffMatrix = np.matlib.repmat(coeff, l_q, 1)

        # Compute the dot product between the symmetry axis of the Watson's distribution
        # and the gradient direction
        #
        # For numerical reasons, cosTheta might not always be between -1 and 1
        # Due to round off errors, individual gradient vectors in grad_dirs and the
        # fibredir are never exactly normal.  When a gradient vector and fibredir are
        # essentially parallel, their dot product can fall outside of -1 and 1.
        #
        # BUT we need make sure it does, otherwise the legendre function call below
        # will FAIL and abort the calculation!!!
        #
        cosTheta = np.dot(grad_dirs,fibredir)
        badCosTheta = abs(cosTheta)>1
        cosTheta[badCosTheta] = cosTheta[badCosTheta]/abs(cosTheta[badCosTheta])

        # Compute the SH values at cosTheta
        sh = np.zeros(coeff.shape)
        shMatrix = np.matlib.repmat(sh, l_q, 1)
        for i in range(7):
            shMatrix[:,i] = np.sqrt((i+1 - .75)/np.pi)
            # legendre function returns coefficients of all m from 0 to l
            # we only need the coefficient corresponding to m = 0
            # WARNING: make sure to input ROW vector as variables!!!
            # cosTheta is expected to be a COLUMN vector.
            tmp = np.zeros((l_q))
            for pol_i in range(l_q):
                tmp[pol_i] = scipy.special.lpmv(0, 2*i, cosTheta[pol_i])
            shMatrix[:,i] = shMatrix[:,i]*tmp

        E = np.sum(lgi*coeffMatrix*shMatrix, 1)
        # with the SH approximation, there will be no guarantee that E will be positive
        # but we need to make sure it does!!! replace the negative values with 10% of
        # the smallest positive values
        E[E<=0] = np.min(E[E>0])*0.1
        E = 0.5*E*ePerp
        return E

    def _cyl_neuman_le_par_PGSE(self, d, G, delta, smalldel):
        # Line bellow used in matlab version removed as cyl_neuman_le_par_PGSE is called from synth_meas_watson_SH_cyl_neuman_PGSE which already casts x to d, R and kappa -> x replaced by d in arguments
        #d=x[0]

        # Radial wavenumbers
        # GAMMA = 2.675987E8
        modQ = GAMMA*smalldel*G
        modQ_Sq = modQ*modQ

        # diffusion time for PGSE, in a matrix for the computation below.
        difftime = (delta-smalldel/3)

        # Parallel component
        LE =-modQ_Sq*difftime*d

        # Compute the Jacobian matrix
        #if(nargout>1)
        #    % dLE/d
        #    J = -modQ_Sq*difftime
        #end
        return LE

    def _cyl_neuman_le_perp_PGSE(self, d, R, G):
        # When R=0, no need to do any calculation
        if (R == 0.00):
            LE = np.zeros(G.shape) # np.size(R) = 1
            return LE
        else:
            ERROR( '"cyl_neuman_le_perp_PGSE" not yet validated for non-zero values' )

    def _legendre_gaussian_integral(self, Lpmp, n):
        if n > 6:
            ERROR( 'The maximum value for n is 6, which corresponds to the 12th order Legendre polynomial' )
        exact = Lpmp>0.05
        approx = Lpmp<=0.05

        mn = n + 1

        I = np.zeros((len(Lpmp),mn))
        sqrtx = np.sqrt(Lpmp[exact])
        I[exact,0] = np.sqrt(np.pi)*scipy.special.erf(sqrtx)/sqrtx
        dx = 1.0/Lpmp[exact]
        emx = -np.exp(-Lpmp[exact])
        for i in range(1,mn):
            I[exact,i] = emx + (i-0.5)*I[exact,i-1]
            I[exact,i] = I[exact,i]*dx

        # Computing the legendre gaussian integrals for large enough Lpmp
        L = np.zeros((len(Lpmp),n+1))
        for i in range(n+1):
            if i == 0:
                L[exact,0] = I[exact,0]
            elif i == 1:
                L[exact,1] = -0.5*I[exact,0] + 1.5*I[exact,1]
            elif i == 2:
                L[exact,2] = 0.375*I[exact,0] - 3.75*I[exact,1] + 4.375*I[exact,2]
            elif i == 3:
                L[exact,3] = -0.3125*I[exact,0] + 6.5625*I[exact,1] - 19.6875*I[exact,2] + 14.4375*I[exact,3]
            elif i == 4:
                L[exact,4] = 0.2734375*I[exact,0] - 9.84375*I[exact,1] + 54.140625*I[exact,2] - 93.84375*I[exact,3] + 50.2734375*I[exact,4]
            elif i == 5:
                L[exact,5] = -(63./256.)*I[exact,0] + (3465./256.)*I[exact,1] - (30030./256.)*I[exact,2] + (90090./256.)*I[exact,3] - (109395./256.)*I[exact,4] + (46189./256.)*I[exact,5]
            elif i == 6:
                L[exact,6] = (231./1024.)*I[exact,0] - (18018./1024.)*I[exact,1] + (225225./1024.)*I[exact,2] - (1021020./1024.)*I[exact,3] + (2078505./1024.)*I[exact,4] - (1939938./1024.)*I[exact,5] + (676039./1024.)*I[exact,6]

        # Computing the legendre gaussian integrals for small Lpmp
        x2=np.power(Lpmp[approx],2)
        x3=x2*Lpmp[approx]
        x4=x3*Lpmp[approx]
        x5=x4*Lpmp[approx]
        x6=x5*Lpmp[approx]
        for i in range(n+1):
            if i == 0:
                L[approx,0] = 2 - 2*Lpmp[approx]/3 + x2/5 - x3/21 + x4/108
            elif i == 1:
                L[approx,1] = -4*Lpmp[approx]/15 + 4*x2/35 - 2*x3/63 + 2*x4/297
            elif i == 2:
                L[approx,2] = 8*x2/315 - 8*x3/693 + 4*x4/1287
            elif i == 3:
                L[approx,3] = -16*x3/9009 + 16*x4/19305
            elif i == 4:
                L[approx,4] = 32*x4/328185
            elif i == 5:
                L[approx,5] = -64*x5/14549535
            elif i == 6:
                L[approx,6] = 128*x6/760543875
        return L

    def _watson_SH_coeff(self, kappa):
        if isinstance(kappa,np.ndarray):
            ERROR( '"watson_SH_coeff()" not implemented for multiple kappa input yet' )

        # In the scope of AMICO only a single value is used for kappa
        n = 6

        C = np.zeros((n+1))
        # 0th order is a constant
        C[0] = 2*np.sqrt(np.pi)

        # Precompute the special function values
        sk = np.sqrt(kappa)
        sk2 = sk*kappa
        sk3 = sk2*kappa
        sk4 = sk3*kappa
        sk5 = sk4*kappa
        sk6 = sk5*kappa
        sk7 = sk6*kappa
        k2 = np.power(kappa,2)
        k3 = k2*kappa
        k4 = k3*kappa
        k5 = k4*kappa
        k6 = k5*kappa
        k7 = k6*kappa

        erfik = scipy.special.erfi(sk)
        ierfik = 1/erfik
        ek = np.exp(kappa)
        dawsonk = 0.5*np.sqrt(np.pi)*erfik/ek

        if kappa > 0.1:

            # for large enough kappa
            C[1] = 3*sk - (3 + 2*kappa)*dawsonk
            C[1] = np.sqrt(5)*C[1]*ek
            C[1] = C[1]*ierfik/kappa

            C[2] = (105 + 60*kappa + 12*k2)*dawsonk
            C[2] = C[2] -105*sk + 10*sk2
            C[2] = .375*C[2]*ek/k2
            C[2] = C[2]*ierfik

            C[3] = -3465 - 1890*kappa - 420*k2 - 40*k3
            C[3] = C[3]*dawsonk
            C[3] = C[3] + 3465*sk - 420*sk2 + 84*sk3
            C[3] = C[3]*np.sqrt(13*np.pi)/64/k3
            C[3] = C[3]/dawsonk

            C[4] = 675675 + 360360*kappa + 83160*k2 + 10080*k3 + 560*k4
            C[4] = C[4]*dawsonk
            C[4] = C[4] - 675675*sk + 90090*sk2 - 23100*sk3 + 744*sk4
            C[4] = np.sqrt(17)*C[4]*ek
            C[4] = C[4]/512/k4
            C[4] = C[4]*ierfik

            C[5] = -43648605 - 22972950*kappa - 5405400*k2 - 720720*k3 - 55440*k4 - 2016*k5
            C[5] = C[5]*dawsonk
            C[5] = C[5] + 43648605*sk - 6126120*sk2 + 1729728*sk3 - 82368*sk4 + 5104*sk5
            C[5] = np.sqrt(21*np.pi)*C[5]/4096/k5
            C[5] = C[5]/dawsonk

            C[6] = 7027425405 + 3666482820*kappa + 872972100*k2 + 122522400*k3  + 10810800*k4 + 576576*k5 + 14784*k6
            C[6] = C[6]*dawsonk
            C[6] = C[6] - 7027425405*sk + 1018467450*sk2 - 302630328*sk3 + 17153136*sk4 - 1553552*sk5 + 25376*sk6
            C[6] = 5*C[6]*ek
            C[6] = C[6]/16384/k6
            C[6] = C[6]*ierfik

        # for very large kappa
        if kappa>30:
            lnkd = np.log(kappa) - np.log(30)
            lnkd2 = lnkd*lnkd
            lnkd3 = lnkd2*lnkd
            lnkd4 = lnkd3*lnkd
            lnkd5 = lnkd4*lnkd
            lnkd6 = lnkd5*lnkd
            C[1] = 7.52308 + 0.411538*lnkd - 0.214588*lnkd2 + 0.0784091*lnkd3 - 0.023981*lnkd4 + 0.00731537*lnkd5 - 0.0026467*lnkd6
            C[2] = 8.93718 + 1.62147*lnkd - 0.733421*lnkd2 + 0.191568*lnkd3 - 0.0202906*lnkd4 - 0.00779095*lnkd5 + 0.00574847*lnkd6
            C[3] = 8.87905 + 3.35689*lnkd - 1.15935*lnkd2 + 0.0673053*lnkd3 + 0.121857*lnkd4 - 0.066642*lnkd5 + 0.0180215*lnkd6
            C[4] = 7.84352 + 5.03178*lnkd - 1.0193*lnkd2 - 0.426362*lnkd3 + 0.328816*lnkd4 - 0.0688176*lnkd5 - 0.0229398*lnkd6
            C[5] = 6.30113 + 6.09914*lnkd - 0.16088*lnkd2 - 1.05578*lnkd3 + 0.338069*lnkd4 + 0.0937157*lnkd5 - 0.106935*lnkd6
            C[6] = 4.65678 + 6.30069*lnkd + 1.13754*lnkd2 - 1.38393*lnkd3 - 0.0134758*lnkd4 + 0.331686*lnkd5 - 0.105954*lnkd6

        if kappa <= 0.1:
            # for small kappa
            C[1] = 4/3*kappa + 8/63*k2
            C[1] = C[1]*np.sqrt(np.pi/5)

            C[2] = 8/21*k2 + 32/693*k3
            C[2] = C[2]*(np.sqrt(np.pi)*0.2)

            C[3] = 16/693*k3 + 32/10395*k4
            C[3] = C[3]*np.sqrt(np.pi/13)

            C[4] = 32/19305*k4
            C[4] = C[4]*np.sqrt(np.pi/17)

            C[5] = 64*np.sqrt(np.pi/21)*k5/692835

            C[6] = 128*np.sqrt(np.pi)*k6/152108775
        return C

    
    # extra-cellular signal
    def _synth_meas_watson_hindered_diffusion_PGSE(self, x, grad_dirs, G, delta, smalldel, fibredir):
        dPar = x[0]
        dPerp = x[1]
        kappa = x[2]

        # get the equivalent diffusivities
        dw = self._watson_hindered_diffusion_coeff(dPar, dPerp, kappa)

        xh = np.array([dw[0], dw[1]])

        E = self._synth_meas_hindered_diffusion_PGSE(xh, grad_dirs, G, delta, smalldel, fibredir)
        return E

    def _watson_hindered_diffusion_coeff(self, dPar, dPerp, kappa):
        dw = np.zeros(2)
        dParMdPerp = dPar - dPerp

        if kappa < 1e-5:
            dParP2dPerp = dPar + 2.*dPerp
            k2 = kappa*kappa
            dw[0] = dParP2dPerp/3.0 + 4.0*dParMdPerp*kappa/45.0 + 8.0*dParMdPerp*k2/945.0
            dw[1] = dParP2dPerp/3.0 - 2.0*dParMdPerp*kappa/45.0 - 4.0*dParMdPerp*k2/945.0
        else:
            sk = np.sqrt(kappa)
            dawsonf = 0.5*np.exp(-kappa)*np.sqrt(np.pi)*scipy.special.erfi(sk)
            factor = sk/dawsonf
            dw[0] = (-dParMdPerp+2.0*dPerp*kappa+dParMdPerp*factor)/(2.0*kappa)
            dw[1] = (dParMdPerp+2.0*(dPar+dPerp)*kappa-dParMdPerp*factor)/(4.0*kappa)
        return dw

    def _synth_meas_hindered_diffusion_PGSE(self, x, grad_dirs, G, delta, smalldel, fibredir):
        dPar=x[0]
        dPerp=x[1]

        # Radial wavenumbers
        # GAMMA = 2.675987E8
        modQ = GAMMA*smalldel*G
        modQ_Sq = np.power(modQ,2.0)

        # Angles between gradient directions and fibre direction.
        cosTheta = np.dot(grad_dirs,fibredir)
        cosThetaSq = np.power(cosTheta,2.0)
        sinThetaSq = 1.0-cosThetaSq

        # b-value
        bval = (delta-smalldel/3.0)*modQ_Sq

        # Find hindered signals
        E=np.exp(-bval*((dPar - dPerp)*cosThetaSq + dPerp))
        return E

    # isotropic signal
    def _synth_meas_iso_GPD(self, d, protocol):
        if protocol['pulseseq'] != 'PGSE' and protocol['pulseseq'] != 'STEAM':
            ERROR( 'synth_meas_iso_GPD() : Protocol %s not translated from NODDI matlab code yet' % protocol['pulseseq'] )

        # GAMMA = 2.675987E8
        modQ = GAMMA*protocol['smalldel'].transpose()*protocol['gradient_strength'].transpose()
        modQ_Sq = np.power(modQ,2)
        difftime = protocol['delta'].transpose()-protocol['smalldel']/3.0
        return np.exp(-difftime*modQ_Sq*d)

    # schemefile
    def _scheme2noddi(self, scheme):
        protocol = {}
        protocol['pulseseq'] = 'PGSE'
        protocol['schemetype'] = 'multishellfixedG'
        protocol['teststrategy'] = 'fixed'
        bval = scheme.b.copy()

        # set total number of measurements
        protocol['totalmeas'] = len(bval)

        # set the b=0 indices
        protocol['b0_Indices'] = np.nonzero(bval==0)[0]
        protocol['numZeros'] = len(protocol['b0_Indices'])

        # find the unique non-zero b-values
        B = np.unique(bval[bval>0])

        # set the number of shells
        protocol['M'] = len(B)
        protocol['N'] = np.zeros((len(B)))
        for i in range(len(B)):
            protocol['N'][i] = np.sum(bval==B[i])

        # maximum b-value in the s/mm^2 unit
        maxB = np.max(B)

        # set maximum G = 40 mT/m
        Gmax = 0.04

        # set smalldel and delta and G
        GAMMA = 2.675987E8
        tmp = np.power(3*maxB*1E6/(2*GAMMA*GAMMA*Gmax*Gmax),1.0/3.0)
        protocol['udelta'] = np.zeros((len(B)))
        protocol['usmalldel'] = np.zeros((len(B)))
        protocol['uG'] = np.zeros((len(B)))
        for i in range(len(B)):
            protocol['udelta'][i] = tmp
            protocol['usmalldel'][i] = tmp
            protocol['uG'][i] = np.sqrt(B[i]/maxB)*Gmax

        protocol['delta'] = np.zeros(bval.shape)
        protocol['smalldel'] = np.zeros(bval.shape)
        protocol['gradient_strength'] = np.zeros(bval.shape)

        for i in range(len(B)):
            tmp = np.nonzero(bval==B[i])
            for j in range(len(tmp[0])):
                    protocol['delta'][tmp[0][j]] = protocol['udelta'][i]
                    protocol['smalldel'][tmp[0][j]] = protocol['usmalldel'][i]
                    protocol['gradient_strength'][tmp[0][j]] = protocol['uG'][i]

        # load bvec
        protocol['grad_dirs'] = scheme.raw[:,0:3].copy()

        # make the gradient directions for b=0's [1 0 0]
        for i in range(protocol['numZeros']):
            protocol['grad_dirs'][protocol['b0_Indices'][i],:] = [1, 0, 0]

        # make sure the gradient directions are unit vectors
        for i in range(protocol['totalmeas']):
            protocol['grad_dirs'][i,:] = protocol['grad_dirs'][i,:]/np.linalg.norm(protocol['grad_dirs'][i,:])

        return protocol