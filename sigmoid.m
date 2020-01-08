function [x] = sigmoid(x)
    x = (x - min(x)) / (max(x) - min(x));
end