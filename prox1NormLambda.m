function z = prox1NormLambda(x,lambda)
  z = max(abs(x) - lambda, zeros(size(x))).*sign(x);
end