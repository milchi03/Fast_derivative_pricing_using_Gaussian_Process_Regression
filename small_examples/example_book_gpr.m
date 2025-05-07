run("G:/Meine Ablage/Ausbildung/BSc. Wirtschaftsmathematik und Statistik/FVM Seminar/Fast_derivative_pricing_using_Gaussian_Process_Regression/gpml-matlab-master/startup.m");
%example as described here: https://gaussianprocess.org/gpml/code/matlab/doc/

x = gpml_randn(0.8, 20, 1);                 % 20 training inputs
y = sin(3*x) + 0.1*gpml_randn(0.9, 20, 1);  % 20 noisy training targets
xs = linspace(-3, 3, 61)';                  % 61 test inputs

meanfunc = [];                    % empty: don't use a mean function
covfunc = @covSEiso;              % Squared Exponental covariance function
likfunc = @likGauss;              % Gaussian likelihood

meanfunc = [];                    % empty: don't use a mean function
covfunc = @covSEiso;              % Squared Exponental covariance function
likfunc = @likGauss;              % Gaussian likelihood

hyp = struct('mean', [], 'cov', [0 0], 'lik', -1);
%hyperparameter optimization
hyp2 = minimize(hyp, @gp, -100, @infGaussLik, meanfunc, covfunc, likfunc, x, y);
[mu s2] = gp(hyp2, @infGaussLik, meanfunc, covfunc, likfunc, x, y, xs);

%plot it
f = [mu + 2*sqrt(s2); flipdim(mu - 2*sqrt(s2), 1)];
fill([xs; flipdim(xs, 1)], f, [7 7 7]/8);
hold on;
plot(xs, mu, 'b-', 'LineWidth', 1.5);       % Predicted mean
plot(x, y, 'k+', 'MarkerSize', 8);          % Training data
plot(xs, sin(3*xs), 'r--', 'LineWidth', 1); % True function

legend('Confidence region', 'Predictive mean', 'Training data', 'True function', 'Location', 'Best');
xlabel('x');
ylabel('y');
title('Gaussian Process Regression with GPML');
grid on;
