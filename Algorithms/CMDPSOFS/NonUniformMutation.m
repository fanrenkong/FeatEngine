function x = NonUniformMutation(x, prob, t)
    r = rand(size(x));
    u = rand(size(x));
    x = x + (r < prob) .* ( ...
        (u <= 0.5) .* ((2.*u).^(1-t)-1) ...
        + (u >= 0.5) .* (1-(2.*(1-u)).^(1-t)));
end