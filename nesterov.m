clear()
A = [1,1/2;1/2,1];
b = [1;-1];
max_gradf = 1.0e-4;
x0 = [5;10];
t = 0.1;
n = 100;
fs = zeros(n,1);
xk = x0;
yk = xk;
tk = 1;

for i=1:n
  gfk = 2*A*yk + b;
  gfk_n2 = norm(gfk);
  if gfk_n2 <= max_gradf
    break
  end
  xk1 = yk - t*gfk;
  tk1 = (1.0 + sqrt(1.0 + 4.0*tk*tk)) / 2.0;
  g = (tk-1)/tk1;
  yk = xk1 + g*(xk1 - xk);
  fk = xk.'*A*xk + b.'*xk;
  fs(i)=fk;
  tk = tk1;
  xk = xk1;
end
fs = fs(1:i-1);
optimizer = xk
steps_to_converge = i-1

rs = conv_rate(fs, -1);
