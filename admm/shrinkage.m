function y = shrinkage(a, k)
  y = (abs(a-k)-abs(a+k))/2 + a;
end
