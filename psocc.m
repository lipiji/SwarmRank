%Particle Swarm Optimization for Classification and Recommender Systems
%Author: Piji Li
%Email: pagelee.sd@gmail.com
%Blog: http://www.zhizhihu.com
%Weibo: http://www.weibo.com/pagecn
function [theta, fit] = psocc(opinion, feature, label)
    c1 = opinion.c1;
    c2 = opinion.c2;
    k = opinion.k;
    phai = c1 + c2;
    X = 2 * k / abs(2 - phai - sqrt(phai^2 - 4*phai));
    m = opinion.swarmSize;
    t = opinion.generations;
    vMax = opinion.vMax;
    %%
    ptcPosition = random('unif',-1,1,m,size(feature, 2));
    ptcVelocity = zeros(m,size(feature, 2));
    ptcBestPos = ptcPosition;
    globalBestPos = zeros(1,size(feature, 2));
    ptcBestFit = ones(m,1)*999999*(-1);
    globalBestFit = -999999;
    
    for i = 1:t
        ptcFit = mapfitness(ptcPosition, feature, label);
        ptcNeedUpdated = find(ptcFit > ptcBestFit);
        ptcBestPos(ptcNeedUpdated,:) = ptcPosition(ptcNeedUpdated,:);
        
        [v, index] = max(ptcFit);
        v = v(1);
        index = index(1);
        newGlobalBestPos = ptcPosition(index, :);

        error = norm(newGlobalBestPos - globalBestPos);
        
        %fprintf('Generation = %d, Fitness = %f, Error = %f\n', i, v, error);
        
        if v > globalBestFit
            globalBestPos = newGlobalBestPos;
            globalBestFit = v;
        end
        
        %更新速度
        w1 = random('unif',0,1,1,1);
        w2 = random('unif',0,1,1,1);
        ptcVelocity = X*(ptcVelocity + c1*w1*(ptcBestPos - ptcPosition) +...
            c2 * w2 * (repmat(globalBestPos,m,1) - ptcPosition));
        ptcVelocity(find(ptcVelocity > vMax)) = vMax;
        ptcVelocity(find(ptcVelocity < -1*vMax)) = -1*vMax;
        
        %更新位置
        ptcPosition = ptcPosition + ptcVelocity;
    end
    theta = globalBestPos';
    fit = globalBestFit;
end

