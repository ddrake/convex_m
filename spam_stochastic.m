% Problem 4 - Stochastic Subgradient Method for Spam filtering
tmp = load('spam_train.csv');
[m,p] = size(tmp);
xx = [tmp,ones(m,1)];
y = load('spam_train_y.csv');

t = 0.005;
n = 10000000;
nd = 100000;        % iterations per display point
xk = x0;
ns = zeros(n/nd,1); % norms of gradient
cs = zeros(n/nd,1); % counts of mis-classified

w = ones(p+1,1);
for i = 1:n
    if mod(i,nd) == 0
        cnz = sum(y .* (xx*w) < 0);
        ns(i) = normw;
        cs(i) = cnz;
    end
    j = randi([1,m]);
    if 1 - y(j) .* xx(j,:)*w > 0
        v = -y(j) .* xx(j,:);
        w = w - t*v.';
        normw = norm(w);
    end 
end

tmp = load('spam_test.csv');
[m,p] = size(tmp);
xx = [tmp,ones(m,1)];
y = load('spam_test_y.csv');

ytest = xx*w;
matches = sum(ytest .* y > 0);
optimizer = w'
percent_correct = 100*matches/size(ytest,1);
percent_correct

