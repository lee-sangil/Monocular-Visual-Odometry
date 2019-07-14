function E = eight_point_essential_hypothesis(obj, x1, x2)

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