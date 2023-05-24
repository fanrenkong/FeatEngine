function PopDec = realInit(N, lb, ub)
    % Initialize population by real encoding
    PopDec = unifrnd(repmat(lb, N, 1), repmat(ub, N, 1));
end