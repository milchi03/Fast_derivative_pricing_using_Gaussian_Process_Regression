from numpy import log,sqrt,exp
from scipy.stats import norm
from QuantLib import *

T = Date(15,1,2016)
S = 100
K = 90
s = 0.20 
q = 0
option_type = Option.Call

r = 0.01
day_count = Actual365Fixed()
calendar = UnitedStates(UnitedStates.NYSE)

t = Date(8, 5, 2015)
Settings.instance().evaluationDate = t

payoff = PlainVanillaPayoff(Option.Call,K)
exercise = EuropeanExercise(T)
european_option = VanillaOption(payoff, exercise)

spot_handle = QuoteHandle(SimpleQuote(S))

flat_ts = YieldTermStructureHandle(FlatForward(t,r,day_count))
dividend_yield = YieldTermStructureHandle(FlatForward(t,q,day_count))
flat_vol_ts = BlackVolTermStructureHandle(BlackConstantVol(t,calendar,s,day_count))

GBM = BlackScholesMertonProcess(spot_handle,dividend_yield, flat_ts, flat_vol_ts)
                                           
european_option.setPricingEngine(AnalyticEuropeanEngine(GBM))
bs_price = european_option.NPV()
print("c=",bs_price)

# elementary Black-Scholes for comparison

ta = (T-t)/365.
d1 = (log(S/K)+(r+s**2/2)*ta)/(s*sqrt(ta))
d2 = (log(S/K)+(r-s**2/2)*ta)/(s*sqrt(ta))
N1 = norm.cdf(d1)
N2 = norm.cdf(d2)
myc = S*N1-exp(-r*ta)*K*N2
print("myc=",myc)

# c= 12.953922942968221
# myc= 12.953922942968205