function a = idx_mat2vec(A, sz)

a = (A(2,:)-1)*sz(1) + A(1,:);