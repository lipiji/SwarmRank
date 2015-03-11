%Particle Swarm Optimization for Classification
%Author: Piji Li
%Email: pagelee.sd@gmail.com
%Blog: http://www.zhizhihu.com
%Weibo: http://www.weibo.com/pagecn

clc;
clear;
%%
opinion.swarmSize = 50;
opinion.generations = 100;
opinion.neighborhoodSize = 10;
opinion.c1 = 2.05;
opinion.c2 = 2.05;
opinion.k = 1;
opinion.vMax = 2;
%%

heart_scale = load('./data/heart_scale.mat');
feature = heart_scale.heart_scale_inst;
label = heart_scale.heart_scale_label;

feature = max_min_norm(feature, 2);

[theta, fit] = psocc(opinion, feature, label)
[theta, fit] = logistic_regression(feature, label)