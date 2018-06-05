clear all
use_saved = true;
m = 100;
n = 20;
if use_saved
  load("subgradientA.mat");
  load("subgradientB.mat");
else
  A = rand(m,n);
  b = rand(m,1);
  save("subgradientA.mat","A");
  save("subgradientB.mat","b");
end

Ata = A.'*A;
Atb = A.'*b;

lambda = 10;
nsteps = 1000;
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
