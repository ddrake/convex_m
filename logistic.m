clear()
tmp = load("LogisticData.txt");
m = size(tmp)(2);
xx = [tmp(1,:).',ones(m,1)];
y = tmp(2,:).';

max_gradf = 1.0e-4;
x0 = [1;1];
t = 0.01;
n = 10000
fs = zeros(n,1);
xk = x0;

for i=1:n
  gfk = xx.'*((1.0 ./ (1+exp(-xx*xk))) - y);
  gfk_n2 = norm(gfk);
  if gfk_n2 <= max_gradf
    break
  end
  xk = xk - t*gfk;
  fk = sum(log(1 + exp(xx*xk)) - y.*xx*xk);
  fs(i)=fk;
end
fs = fs(1:i-1);
