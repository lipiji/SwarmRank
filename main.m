%% Swarming to Rank for Recommender Systems
%Particle Swarm Optimization for Classification and Recommender Systems
%Author: Piji Li
%Email: pagelee.sd@gmail.com
%Blog: http://www.zhizhihu.com
%Weibo: http://www.weibo.com/pagecn
clc;
clear;
%% 并行计算，多线程
matlabpool(2);  % 有2个核
try
%%
opinion.swarmSize = 50;
opinion.generations = 100;
opinion.c1 = 2.05;
opinion.c2 = 2.05;
opinion.k = 1;
opinion.vMax = 2;
%%
k =46;
n4Top = [1 5 10 20]; %TopN N=1,5,10,20
s4User = 10;
[oldUID, newUID]=textread('./data/uid.dat','%s %d');
[oldIID, newIID]=textread('./data/iid.dat','%s %d');

numU = length(newUID);
numI = length(newIID);

%%
m_train = load('./data/m_train.dat');
m_train = load_sparse_matrix(m_train, numU, numI);
%SVD矩阵分解
[U,S,V] = svds(m_train, k);
clear m_train;
train_set = load('./data/train_set.dat');
train_sample = unique(train_set(:,2));

test_set = load('./data/test_set.dat');
test_sample = unique(test_set(:,2));

%% 每个user用粒子群来训练模型，其实也可以用LR、SVM等来训练模型
topN = [];
parfor i = 1 : numU
    % 构建User i的训练集和测试集
    fprintf('User %d...\n', i);
    test_set_u = test_set(find(test_set(:,1)==i),:);
    [asort index] = sort(test_set_u(:,3),'descend');
    %测试集中取出的样本，可以取1个，也可以取多个，
    %为了度量更准确更平，这里取s4User个
    test_set_u = test_set_u(index(1:s4User),:);
    
    %除了上面的样本，还要随机取一些其他样本，计算TopN用
    test_random = setdiff(train_sample, test_set_u(:,2)); %负样本
    test_random = test_random(randperm(length(test_random))); %随机排序
    test_random = test_random(1:100); %取100个；

    %构建训练集
    train_set_u = train_set(find(train_set(:,1)==i),:);
    train_set_u(:,3) = 1; %正样本
    
    % 平衡样本，防止样本不平衡
    numPositive = size(train_set_u, 1);
    numNegative = numPositive;
    if numNegative < 100
        numNegative = 100;
    end
    negative_sample = setdiff(train_sample, train_set_u(:,2));
    negative_sample = negative_sample(randperm(length(negative_sample)));
    for j = 1 : numNegative
        train_set_u = [train_set_u;[i negative_sample(j) -1]];
    end
    
    % 每个item的特征
    train_label_u = train_set_u(:,3);
    train_feature_u = V(train_set_u(:,2),:);
    [model, fit] = psocc(opinion, train_feature_u, train_label_u);
    %[model, fit] = logistic_regression(train_feature_u, train_label_u)
    
    %测试Top-N
    test_random_feature = V(test_random,:);
    test_random_decision = psopredict(model, test_random_feature);
    test_random_decision = sort(test_random_decision, 'descend');
    result_u  = [];
    for s = 1 : s4User
        test_feature_s = V(test_set_u(s,2),:);
        test_decision_s = psopredict(model, test_feature_s);
        for n = 1 : length(n4Top)
            if test_decision_s < test_random_decision(n4Top(n))
                result_u(s, n) = 0;
            else
                result_u(s, n) = 1;
            end
        end
    end
    
    result_u = mean(result_u);
    
    topN(i,:) = result_u;
end

topN = sum(topN) / size(topN,1);
for i = 1 : length(topN)
    fprintf('Recall on Top %d = %f.\n', n4Top(i), topN(i));
end
matlabpool close; %如果中间意外退出，需要手动关闭线程池

catch
    disp('意外结束: 异常或用户终止.');
    matlabpool close; %如果中间意外退出，需要手动关闭线程池
end


