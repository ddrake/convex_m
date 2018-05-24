% check a sequence for linear convergence
function rs = conv_rate( ys, limit )
    m = size(ys,1);
    rs = (ys(2:m) - limit) ./ (ys(1:m-1) - limit);
    rs = rs(2:m-1);
    plot(rs)
    title("Linear Convergence Rate")
end

