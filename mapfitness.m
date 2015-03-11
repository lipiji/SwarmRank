function fit = mapfitness(theta, feature, label)
    n = length(label); % for P@n and MAP:https://www.kaggle.com/wiki/MeanAveragePrecision
    fit = zeros(size(theta, 1), 1);
    predict = sigmf(feature * theta', [1, 0]); % sigmoid function like LR
    for i = 1 : size(predict, 2)
        [v, index] = sort(predict(:,i), 'descend');
        v(find(v < 0.5)) = -1;
        v(find(v >= 0.5)) = 1;
        tLabel = label(index);
        fit(i, 1) = (sum(v(1:n) == tLabel(1:n))) / n;
    end
end

