function decision = psopredict(theta, feature)
    decision = sigmf(feature * theta, [1, 0]); % sigmoid function like LR
end

