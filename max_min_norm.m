function A = max_min_norm(A, n)
    if n == 1
        for i = 1 : size(A, 1)
            A(i,:) = (A(i,:) - min(A(i,:))) / (max(A(i,:)) - min(A(i,:)));
        end
    end
    if n == 2
        for i = 1 : size(A, 2)
            A(:,i) = (A(:,i) - min(A(:,i))) / (max(A(:,i)) - min(A(:,i)));
        end
    end
end
