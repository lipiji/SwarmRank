function mat = load_sparse_matrix(sp_mat, r, c )
mat = zeros(r, c);
for i = 1: size(sp_mat, 1)
    mat(sp_mat(i,1), sp_mat(i,2)) = sp_mat(i,3);
end

