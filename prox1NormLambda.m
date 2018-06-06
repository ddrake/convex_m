%% Function prox1NormLambda
% vectorized function for computing the prox for the Lasso Problem
function z = prox1NormLambda(x,lambda)
  z = max(abs(x) - lambda, zeros(size(x))).*sign(x);
end