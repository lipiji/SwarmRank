%% Swarming to Rank for Recommender Systems
%Particle Swarm Optimization for Classification and Recommender Systems
%Author: Piji Li
%Email: pagelee.sd@gmail.com
%Blog: http://www.zhizhihu.com
%Weibo: http://www.weibo.com/pagecn
clc;
clear;
%% ���м��㣬���߳�
matlabpool(2);  % ��2����
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
%SVD����ֽ�
[U,S,V] = svds(m_train, k);
clear m_train;
train_set = load('./data/train_set.dat');
train_sample = unique(train_set(:,2));

test_set = load('./data/test_set.dat');
test_sample = unique(test_set(:,2));

%% ÿ��user������Ⱥ��ѵ��ģ�ͣ���ʵҲ������LR��SVM����ѵ��ģ��
topN = [];
parfor i = 1 : numU
    % ����User i��ѵ�����Ͳ��Լ�
    fprintf('User %d...\n', i);
    test_set_u = test_set(find(test_set(:,1)==i),:);
    [asort index] = sort(test_set_u(:,3),'descend');
    %���Լ���ȡ��������������ȡ1����Ҳ����ȡ�����
    %Ϊ�˶�����׼ȷ��ƽ������ȡs4User��
    test_set_u = test_set_u(index(1:s4User),:);
    
    %�����������������Ҫ���ȡһЩ��������������TopN��
    test_random = setdiff(train_sample, test_set_u(:,2)); %������
    test_random = test_random(randperm(length(test_random))); %�������
    test_random = test_random(1:100); %ȡ100����

    %����ѵ����
    train_set_u = train_set(find(train_set(:,1)==i),:);
    train_set_u(:,3) = 1; %������
    
    % ƽ����������ֹ������ƽ��
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
    
    % ÿ��item������
    train_label_u = train_set_u(:,3);
    train_feature_u = V(train_set_u(:,2),:);
    [model, fit] = psocc(opinion, train_feature_u, train_label_u);
    %[model, fit] = logistic_regression(train_feature_u, train_label_u)
    
    %����Top-N
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
matlabpool close; %����м������˳�����Ҫ�ֶ��ر��̳߳�

catch
    disp('�������: �쳣���û���ֹ.');
    matlabpool close; %����м������˳�����Ҫ�ֶ��ر��̳߳�
end


