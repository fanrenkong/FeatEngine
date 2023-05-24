function x = UniformMutation(x, prob)
    r = rand(size(x));
    x = x + (r < prob) .* (unifrnd(0, 1, size(x)) - 0.5);
end

