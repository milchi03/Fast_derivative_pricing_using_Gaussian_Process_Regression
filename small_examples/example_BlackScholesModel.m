% Initialize the GPML toolbox
run("G:/Meine Ablage/Ausbildung/BSc. Wirtschaftsmathematik und Statistik/FVM Seminar/Fast_derivative_pricing_using_Gaussian_Process_Regression/gpml-matlab-master/startup.m");
pkg load statistics

% Set Black-Scholes parameters for the sample data
sample_size = 100;
S = 100;                                   % Single Stock price (scalar)
K = 100;                                   % Strike price (constant)
T = 0.5 + 1 * rand(sample_size, 1);        % Time to maturity
r = 0.05;                                  % Risk-free rate
sigma = 0.15 + 0.1 * rand(sample_size, 1); % Volatility

% Calculate option prices using Black-Scholes formula
d1 = (log(S ./ K) + (r + sigma.^2 / 2) .* T) ./ (sigma .* sqrt(T));
d2 = d1 - sigma .* sqrt(T);
y = S .* normcdf(d1) - K .* exp(-r .* T) .* normcdf(d2);  % Option prices

% Combine inputs into one matrix for GPR
x = [T, sigma]; % S is constant, removed from x as it's no longer variable

% Set up Gaussian Process Regression
meanfunc = [];
covfunc = @covSEiso;
likfunc = @likGauss;
hyp = struct('mean', [], 'cov', [log(1), log(1)], 'lik', log(0.1));

% Optimize hyperparameters
hyp2 = minimize(hyp, @gp, -500, @infGaussLik, meanfunc, covfunc, likfunc, x, y);

% Define 10 new test values within the training range
T_test = 0.5 + 1 * rand(10, 1);                  % Random T values within range
sigma_test = 0.15 + 0.1 * rand(10, 1);           % Random sigma values within range
x_test = [T_test, sigma_test];

% Predict option prices for the test points
[mu_test, ~] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, x_test);

% Create a grid for T and sigma
[T_grid, sigma_grid] = meshgrid(linspace(0.5, 1.5, 30), linspace(0.15, 0.25, 30));

% Set a constant stock price for the surface plot
S_const = 100; % Stock price is fixed

% Calculate the option prices over the grid using Black-Scholes formula
d1_grid = (log(S_const ./ K) + (r + sigma_grid.^2 / 2) .* T_grid) ./ (sigma_grid .* sqrt(T_grid));
d2_grid = d1_grid - sigma_grid .* sqrt(T_grid);
price_grid = S_const .* normcdf(d1_grid) - K .* exp(-r .* T_grid) .* normcdf(d2_grid);

% Generate the 3D plot
figure;
hold on;
scatter3(T_test, sigma_test, mu_test, 'r', 'filled'); % Predicted prices for test data
surf(T_grid, sigma_grid, price_grid, 'FaceAlpha', 0.5, 'EdgeColor', 'none'); % Black-Scholes surface

% Set labels, title, and legend explicitly
xlabel('Maturity (T)', 'FontSize', 12);
ylabel('Volatility (\sigma)', 'FontSize', 12);
zlabel('Option Price', 'FontSize', 12);
title('Comparison of Black-Scholes Model Surface and GP Predictions for Test Data', 'FontSize', 14);
legend('GP Predicted Prices (Test Data)', 'Black-Scholes Surface');
grid on;

