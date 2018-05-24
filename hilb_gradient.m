clear()
H = hilb(5)
max_gradf = 1.0e-2;
x0 = [1.;2.;3.;4.;5.];
t = 0.1;
fs = zeros(100,1);
xk = x0;

for i=1:100
  gfk = 2*H*xk;
  gfk_n2 = norm(gfk);
  if gfk_n2 <= max_gradf
    break
  end
  xk = xk - t*gfk;
  fk = xk.'*H*xk;
  fs(i)=fk;
end

