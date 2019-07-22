function h = test_essentialmatrix

n = 100;
addpath(genpath('../5-point-algorithm-MATLAB-master/'));

for i = 1:500
w_ = [0.1*rand() 1 0.1*rand()]';
w = w_/norm(w_);
theta = 0.1*rand;
R_true = expm(skew(w)*theta);
t_ = -0.5+rand(3,1);
t_true = t_ / norm(t_);

X1 = 100*rand(3,n);
x1 = X1 ./ X1(3,:);
X2 = R_true*X1+t_true;
x2 = X2 ./ X2(3,:);

cv_featurePts1 = num2cell(x1(1:2,:)',2);
cv_featurePts2 = num2cell(x2(1:2,:)',2);

% E = cv.findEssentialMat(cv_featurePts1,cv_featurePts2,'Threshold',1e-7);

ransacCoef.iterMax = 1e4;
ransacCoef.minPtNum = 8;
ransacCoef.thInlrRatio = 0.999;
ransacCoef.thDist = 1e-7;
ransacCoef.thDistOut = 1e-7;
% ransacCoef.funcFindF = @eight_point_essential_hypothesis;
ransacCoef.funcFindF = @fivePoint;
ransacCoef.funcDist = @essential_model_error;

[E, inlier, ~] = ransac(x1, x2, ransacCoef);

[U, ~, V] = svd(E);
if det(U) <  0
	U(:,3) = -U(:,3);
end
if det(V) < 0
	V(:,3) = -V(:,3);
end

% Extract rotational and translational movement
W = [0 -1 0; 1 0 0; 0 0 1];
R_vec{1} = U*W*V.';
R_vec{2} = U*W*V.';
R_vec{3} = U*W.'*V.';
R_vec{4} = U*W.'*V.';
t_vec{1} = U(:,3);
t_vec{2} = -U(:,3);
t_vec{3} = U(:,3);
t_vec{4} = -U(:,3);

[R, t, inlier] = verify_solutions(x1, x2, R_vec, t_vec);

error(i) = norm([R_true t_true]-[R t]);

end

figure;
subplot(121);histogram(error, 'BinEdge', linspace(0,1e-10,20));
subplot(122);histogram(error, 'BinEdge', linspace(0,1,20));

end

function E = eight_point_essential_hypothesis(x1, x2)

a = [x1(1,:).*x2(1,:);
	x1(1,:).*x2(2,:);
	x1(1,:).*x2(3,:);
	x1(2,:).*x2(1,:);
	x1(2,:).*x2(2,:);
	x1(2,:).*x2(3,:);
	x1(3,:).*x2(1,:);
	x1(3,:).*x2(2,:);
	x1(3,:).*x2(3,:);];

a_t = a.';
[~, ~, V] = svd(a_t.'*a_t);
E = reshape(V(:,end), 3, 3);

% re-enforce rank 2
[U, ~, V] = svd(E);
if det(U) <  0
	U(:,3) = -U(:,3);
end
if det(V) < 0
	V(:,3) = -V(:,3);
end

% sigma = ( S(1,1) + S(2,2) ) / 2;
S = diag([1, 1, 0]);
E = U * S * V.';

end

function dist = essential_model_error(E, x1, x2)

% sampson error
Ex1 = E*x1;
Etx2 = E.'*x2;

% for i = 1:n
% 	d(i) = ( uv2(:,i).'*F*uv1(:,i) ).^2 ./ ( Fu1(i)^2 + Fv1(i)^2 + Ftu2(i)^2 + Ftv2(i)^2 );
% end

% d = diag(uv2.'*F*uv1).^2 ./ sum( (F(1:2,:)*uv1).^2 + (F(:,1:2).'*uv2).^2 ).';

dist = diag(x2.'*E*x1).^2 ./ ( Ex1(1,:).^2 + Ex1(2,:).^2 + Etx2(1,:).^2 + Etx2(2,:).^2 + eps ).';

end

function [R, t, max_inlier] = verify_solutions( x1, x2, R_vec, t_vec )

n = size(x1,2);

% Find reasonable rotation and translational vector
max_num = 0;
for i = 1:length(R_vec)
	R1 = R_vec{i};
	t1 = t_vec{i};
	
	m_ = cell(n, 1);
	t_ = zeros(3*n, 1);
	for j = 1:n
		m_{j} = skew(x2(:,j))*R1*x1(:,j);
		t_(3*(j-1)+1:3*j) = skew(x2(:,j))*t1;
	end
	M = blkdiag(m_{:});
	M(:, n+1) = t_;
	
	[~,~,V] = svd(M.'*M);
	lambda_prev = V(1:end-1,end).'/(V(end,end));
	
	lambdax_curr = bsxfun(@plus, bsxfun(@times, lambda_prev, R1*x1), t1);
	lambda_curr = lambdax_curr(3,:);
	
	inlier = find(lambda_prev > 0 & lambda_curr > 0);
	
	if length(inlier) > max_num
		max_num = length(inlier);
		max_inlier = inlier;
		
		R = R1;
		t = t1;
	end
end

end