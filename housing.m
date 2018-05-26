% Problem 3 - Housing Price Regression
clear()
tmp = load('Housing.txt');
m = size(tmp,1);
xx = [tmp(:,1:2),ones(m,1)];
y = tmp(:,3);

% normalize for stability and improved convergence
xmax = max(xx);
ymax = max(y);
xx = xx ./ xmax;
y = y ./ ymax;
xtx = xx.'*xx;
xty = xx.'*y;
yty = y.'*y;

max_gradf = 1.0e-5;
x0 = [1;1;1];
t = 0.01;
n = 2000;
fs = zeros(n,1);
xk = x0;

for i=1:n
  gfk = 2*(xtx*xk - xty);
  gfk_n2 = norm(gfk);
  if gfk_n2 <= max_gradf
    break
  end
  xk = xk - t*gfk;
  fk = xk.'*xtx*xk - 2*xty.'*xk + yty;
  fs(i)=fk;
end
fs = fs(1:i-1);
% change back to original variables
xk = diag(1 ./ xmax)*(xk)*ymax;

optimizer = xk
steps_to_converge = i-1

price = [2080.,4,1]*xk


