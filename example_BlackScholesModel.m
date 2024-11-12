run("G:/Meine Ablage/Ausbildung/Wirtschaftsmathematik und Statistik/Seminar GPA for fast derivative pricing/Fast_derivative_pricing_using_Gaussian_Process_Regression/gpml-matlab-master/startup.m");

% Set Black-Scholes parameters for 1000 training inputs
S = 50 + 100 * rand(1000, 1);        % Stock prices
K = 100;                             % Strike price (constant)
T = 0.5 + 1 * rand(1000, 1);         % Time to maturity
r = 0.05;                            % Risk-free rate
sigma = 0.15 + 0.1 * rand(1000, 1);  % Volatility

% Calculate option prices using Black-Scholes formula
d1 = (log(S/K) + (r + sigma.^2 / 2) .* T) ./ (sigma .* sqrt(T));
d2 = d1 - sigma .* sqrt(T);
y = S .* normcdf(d1) - K * exp(-r * T) .* normcdf(d2);

% Combine inputs into one matrix for GPR
x = [S, T, sigma];

% Set up Gaussian Process Regression
meanfunc = [];
covfunc = @covSEiso;
likfunc = @likGauss;
hyp = struct('mean', [], 'cov', [log(1), log(1)], 'lik', log(0.1));

% Optimize hyperparameters
hyp2 = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y);

% Predict the option price for a new option
S_new = 105;
T_new = 1.2;
sigma_new = 0.25;
xs = [S_new, T_new, sigma_new];

% Calculate predicted option price
[mu, s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);

% Plot results
f = [mu + 2 * sqrt(s2); flipud(mu - 2 * sqrt(s2))];
fill([xs; flipud(xs)], f, [7 7 7]/8);
hold on; plot(xs, mu); plot(x, y, '+');
xlabel('Input parameter space');
ylabel('Option price');
title('GP Prediction of Option Prices');

