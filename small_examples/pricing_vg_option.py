#Example from "Option pricing under the variance gamma process" by Fiorani, Page 153
from numpy import log, sqrt, exp
from scipy.stats import norm
from QuantLib import *

# Parameters
T = Date(15, 1, 2016)  # Maturity date
K = 100  # Strike price
r = 0.0514  # Risk-free rate
q = 0.014  # Dividend yield
sigma = 0.20722  # Volatility
theta = -0.22898  # Skewness parameter
nu = 0.50215  # Kurtosis parameter
S = 100  # Current stock price

# Day count and calendar
day_count = Actual365Fixed()
calendar = UnitedStates(UnitedStates.NYSE)

# Evaluation date
t = Date(15, 1, 2015)
Settings.instance().evaluationDate = t

# Option type
option_type = Option.Call

# Payoff and exercise
payoff = PlainVanillaPayoff(option_type, K)
exercise = EuropeanExercise(T)
european_option = VanillaOption(payoff, exercise)

# Market data
spot_handle = QuoteHandle(SimpleQuote(S))
flat_ts = YieldTermStructureHandle(FlatForward(t, QuoteHandle(SimpleQuote(r)), day_count))
dividend_yield = YieldTermStructureHandle(FlatForward(t, QuoteHandle(SimpleQuote(q)), day_count))

# Variance Gamma process
VG_process = VarianceGammaProcess(spot_handle, dividend_yield, flat_ts, sigma, nu, theta)

# Pricing engine
european_option.setPricingEngine(VarianceGammaEngine(VG_process))

# Compute the price
vg_price = european_option.NPV()
print("c =", vg_price) #c