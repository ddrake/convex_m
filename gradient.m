clear()
A = [1,1/2;1/2,1];
b = [1;-1];
max_gradf = 1.0e-4;
x0 = [5;10];
t = 0.1;
fs = zeros(100,1);
xk = x0;

for i=1:100
  gfk = 2*A*xk + b;
  gfk_n2 = norm(gfk);
  if gfk_n2 <= max_gradf
    break
  end
  xk = xk - t*gfk;
  fk = xk.'*A*xk + b.'*xk;
  fs(i)=fk;
end

