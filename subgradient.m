%% Subgradient Algorithm with Lasso Problem
% Save random matrix for comparison of algorithms

%% Generate random matrices or used saved ones for comparison
clear
use_saved = true;
if use_saved
  load("subgradientA.mat");
  load("subgradientB.mat");
  [m,n] = size(A);
else
  m = 100;
  n = 20;
  A = rand(m,n);
  b = rand(m,1);
  save("subgradientA.mat","A");
  save("subgradientB.mat","b");
end

Ata = A.'*A;
Atb = A.'*b;
%% Subgradient Algorithm
% lambda - Weight on 1-norm of x
lambda = 10;
nsteps = 400;
fs = zeros(nsteps,1);
vs = zeros(nsteps,1);
xs = zeros(nsteps,1);
x = zeros(n,1);
t = 0.5/norm(Ata);
for i=1:nsteps
  x = x - t * (Ata*x - Atb + lambda * sign(x));
  f = .5*norm(A*x - b,2)^2 + lambda * norm(x,1);
  fs(i) = f;
  xs(i) = norm(x,1);
end

plot(fs)
title("Subgradient: 1/2 ||A*x - b||^2 + \lambda * ||x||_1")
%conv_rate = (fs(2:nsteps)-fs(nsteps))./(fs(1:nsteps-1)-fs(nsteps));
%conv_rate = mean(conv_rate)
norm_x1 = xs(nsteps)

%% Notes  - Lambda = 1.0 case
% Convergence is rapid at first, should be roughly O(sqrt(1/k))
%
% Increasing lambda reduces the size of the 1-norm of x relative to 
% norm squared of Ax-b.
% 
% We see that higher values of lambda result in jitter in the solution
% values

