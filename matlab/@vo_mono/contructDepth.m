function [X_prev, X_curr, lambda_prev, lambda_curr] = contructDepth( obj, x_prev, x_curr, R, t )

n = size(x_prev,2);
m_ = cell(n, 1);
t_ = zeros(3*n, 1);
for j = 1:n
	m_{j} = skew(x_curr(:,j))*R*x_prev(:,j);
	t_(3*(j-1)+1:3*j) = skew(x_curr(:,j))*t;
end
M = blkdiag(m_{:});
M(:, n+1) = t_;

[~,~,V] = svd(M.'*M);
lambda_prev = V(1:end-1,end).'/(V(end,end));

X_prev = bsxfun(@times, lambda_prev, x_prev);
X_curr = bsxfun(@plus, R*X_prev, t);
lambda_curr = X_curr(3,:);

end