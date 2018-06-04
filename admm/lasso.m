% The lambda parameter allows us to trade off Ax-b error against the size of x
% if lambda is small, the error should be smaller but x will be bigger.
clear()
newmat = true;
% for this implementation to make sense
% we need a tall thin matrix.
m = 2000;
n = 50;
if newmat
  A = rand(m,n);
  b = rand(m,1);
  save('LA.mat','A')
  save('Lb.mat','b')
else
  load('LA.mat')
  load('Lb.mat')
end
rho = 1000;
% note: I think we could decrease rho as lambda is increased and still
% get good convergence.
xns = zeros(6,1);
sqerrs = zeros(6,1);
for j = -3:3
  lambda = 2^j
  z = zeros(n,1);
  u = zeros(n,1);
  % x = A\B solves Ax = B
  % so A\B should be equal to A^-1*B 
  AtArI = A.'*A + rho*eye(n);
  Atb = A.'*b;
  xp = zeros(n,1);
  iters=500;
  ns = zeros(iters,1);
  % note that we must solve a system each step.
  for i = 1:iters
    x = AtArI\(Atb + rho*(z - u));
    z = shrinkage(x + u,lambda/rho);
    u = u + x - z;
    nx1 = norm(x,1);
    ns(i) = nx1;
  end
  plot(ns)
  title("Convergence of norm x")
  sqerr=(A*x-b).'*(A*x-b);
  x1norm = ns(iters);
  sqerrs(j+4) = sqerr;
  xns(j+4) = nx1;
  input('Press any key')
end
plot(sqerrs, xns)
title("Lasso Problem")
xlabel("1/2||Ax - b||^2")
ylabel("||x||_1")