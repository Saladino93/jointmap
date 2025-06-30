import numpy as np
import matplotlib.pyplot as plt
from jointmap import plot
import utils_data as ud

plot.set_default_params()

def getsigma2(claa, noise, fsky = 1.):
    ells = np.arange(len(noise))
    nmodes = (2*ells + 1)*fsky
    total = claa+noise
    errorbar = np.sqrt(2/nmodes)*total
    sigma2 = errorbar**2
    return sigma2

def Abias(bias, claa, noise, Lmax, Lmin = 1, fsky = 1.):
    ells = np.arange(len(noise))
    sigma2 = getsigma2(claa, noise, fsky)
    num = sigma2**-1*bias*claa
    den = claa**2/sigma2
    selection = (ells <= Lmax) & (ells >= Lmin)
    return np.sum(num[selection])/np.sum(den[selection])

def SNR(claa, noise, Lmax, Lmin = 1, fsky = 1.):
    ells = np.arange(len(noise))
    nmodes = (2*ells + 1)*fsky
    total = claa+noise
    num = claa**2*nmodes
    den = 2*total**2
    result = num/den
    selection = (ells <= Lmax) & (ells >= Lmin)
    return np.sqrt(np.sum(result[selection]))

def SNRbire(A, noise, Lmax, Lmin = 1, fsky = 1.):
    ells = np.arange(len(noise))
    nmodes = (2*ells + 1)*fsky
    factor = (ells)*(ells+1)/2/np.pi
    total = A+noise*factor
    num = A**2*nmodes
    den = 2*total**2
    result = num/den
    selection = (ells <= Lmax) & (ells >= Lmin)
    return np.sqrt(np.sum(result[selection]))


colors = plot.COLOR_BLIND_FRIENDLY["wong"]
names = {"s4": "S4-like", "spt": "SPT-3G-like", "so": "SO-like"}

for i, name in enumerate(["s4", "spt", "so"]):
    #name = "s4"
    qty = ud.get_noise_curves(case = name)
    bias = qty["n1ap0"]
    naa = qty["n1aa0"]
    noise = qty["ngg0"]+bias
    claa = qty["claath"]

    fsky = 0.4
    Lmin, Lmax = 1, 100
    Lmaxes = np.arange(1, 1000, 1)
    sigma_A = np.array([SNR(claa, noise, Lmax = Lmax, Lmin = Lmin, fsky = fsky) for Lmax in Lmaxes])
    sigma_A_QE = sigma_A
    #sigma_A = np.array([SNR(claa+naa, noise, Lmax = Lmax, Lmin = Lmin, fsky = fsky) for Lmax in Lmaxes])
    #sigma_A_QE_aa = sigma_A
    bias_A = np.array([Abias(bias, claa, noise, Lmax = Lmax, Lmin = Lmin, fsky = fsky) for Lmax in Lmaxes])
    bias_A_QE = bias_A

    #plt.plot(Lmaxes, sigma_A)
    #plt.plot(Lmaxes, bias_A, ls = "--")

    bias = qty["n1apmax"]
    noise = qty["nggmax"]+bias
    naa = qty["n1aamax"]
    #naaderot = qty["naaderot"]
    sigma_A = np.array([SNR(claa, noise, Lmax = Lmax, Lmin = Lmin, fsky = fsky) for Lmax in Lmaxes])
    #sigma_A_aa = np.array([SNR(claa+naaderot, noise, Lmax = Lmax, Lmin = Lmin, fsky = fsky) for Lmax in Lmaxes])
    bias_A = np.array([Abias(bias, claa, noise, Lmax = Lmax, Lmin = Lmin, fsky = fsky) for Lmax in Lmaxes])

    plt.plot(Lmaxes, sigma_A/sigma_A_QE, color = colors[i], label = names[name])
    #plt.plot(Lmaxes, sigma_A_aa/sigma_A_QE_aa)
    plt.plot(Lmaxes, bias_A/bias_A_QE, ls = "--", color = colors[i])

plt.xscale("log")
plt.legend(fontsize = 16)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.xlabel(r"$L_{\mathrm{max}}$", fontsize = 18)
plt.ylabel(r"$\frac{X_{\mathrm{MAP}}}{X_{\mathrm{QE}}}$", fontsize = 20)
plt.xlim(2, 1000)
plt.savefig(f"comparisons.pdf")
plt.savefig(f"{ud.out_dir}comparisons.pdf")