function [theta, fit] = logistic_regression(feature, label)
% Implementation of Gradient (Steepest) Descent / batch method 
% min 1/2*||WX - Y||^2
% Piji Li

X = feature;
Y = label;

epsilon = 0.0003;
gama = 0.001;

w_old=zeros(size(X,2),1);
k=1;
t = 100;
while 1
    minJ_w(k) = 1/2 * (norm(X*w_old - Y))^2;
    w_new = w_old - gama*(X'*X*w_old - X'*Y);
    %fprintf('The %dth iteration, minJ_w = %f, \n',k,minJ_w(k));
    
    if (k >= t) || (norm(w_new-w_old) < epsilon)
        W_best = w_new;
        break;
    end
    w_old = w_new;
    k=k+1;
end


%% prediction
for i=1:length(Y)
    decesion(i,1) = 1/(exp(-1*(X(i,:)*w_new))+1);
end
pred_label = ones(size(X,1),1)*(-1);
pred_label(find(decesion >= 0.5)) = 1;
accuracy = length(find(pred_label == Y))/length(Y);

theta = W_best;
fit = accuracy;

