% Increasing rho generally pulls the 1-norm of x down,
% but as we increase rho we may need to increase the number of iterations.
% for a matrix with rank 999, nullity 1, any rho > 0.0001 seems to work
%   Increasing rho above 1000 forces us to increase the iterations, but with
%   no noticeable benefit.
% For a matrix with rank 500, nullity 500, we get an improvement 
%   when rho increases from 10 100
% for a matrix with rank 1, nullity 999, there is a noticeable improvement 
%   when rho increases from 1 to 10.
% Conclusion: the larger the null space relative to the rank, the more benefit
%   we may get by increasing rho.
clear()
newmat = false;
% for this implementation to make sense
% we need a short wide matrix.
m = 500;
n = 1000;
if newmat
  A = rand(m,n);
  b = rand(m,1);
  save('BpA.mat','A')
  save('Bpb.mat','b')
else
  load('BpA.mat')
  load('Bpb.mat')
end
rho = 100.0;
z = zeros(n,1);
u = zeros(n,1);
% x = A\B solves Ax = B
% so A\B should be equal to A^-1*B 
IA4 = eye(n) - A.'*((A*A.')\A);
A3B = A.'*((A*A.')\b);
xp = zeros(n,1);
iters=2000;
ns = zeros(iters,1);
% no need to solve a system each step.
for i = 1:iters
  x = IA4*(z - u) + A3B;
  z = shrinkage(x + u, 1./rho);
  u = u + x - z;
  nx1 = norm(x,1);
  ns(i) = nx1;
end
plot(ns)
err=norm(A*x-b)
x1norm = ns(iters)
rho
nullity = n-m