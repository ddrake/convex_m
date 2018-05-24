clear()
A = [1,1/2;1/2,1];
b = [1;-1];
max_gradf = 1.0e-4;
x0 = [5;10];
t = 0.1;
fs = zeros(100,1);
xk = x0;
yk = xk;
tk = 1;

for i=1:100
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
