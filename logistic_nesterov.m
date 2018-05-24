clear()
tmp = load("LogisticData.txt");
m = size(tmp)(2);
xx = [tmp(1,:).',ones(m,1)];
y = tmp(2,:).';

max_gradf = 1.0e-4;
x0 = [1;1];
t = 0.01;
n = 1000
fs = zeros(n,1);
xk = x0;
yk = xk;
tk = 1;

for i=1:n
  gfk = xx.'*((1.0 ./ (1+exp(-xx*yk))) - y);
  gfk_n2 = norm(gfk);
  if gfk_n2 <= max_gradf
    break
  end
  xk1 = yk - t*gfk;
  tk1 = (1.0 + sqrt(1.0 + 4.0*tk*tk)) / 2.0;
  g = (tk-1)/tk1;
  yk = xk1 + g*(xk1 - xk);
  fk = sum(log(1 + exp(xx*xk)) - y.*xx*xk);
  fs(i)=fk;
  tk = tk1;
  xk = xk1;
end
fs = fs(1:i-1);
