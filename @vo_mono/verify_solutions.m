function [R, t] = verify_solutions( obj, R_vec, t_vec )
%% VERIFICATION
% Load parameters from object
K = obj.params.K;
nFeature = obj.nFeature;
nFeature2DInliered = obj.nFeature2DInliered;

% Extract homogeneous 2D point which is inliered with essential constraint
i_inliered = zeros(1, nFeature2DInliered);
j = 1;
uv_prev = zeros(2, nFeature2DInliered);
uv_curr = zeros(2, nFeature2DInliered);
for i = 1:nFeature
	if obj.features(i).is_2D_inliered
		uv_prev(:,j) = obj.features(i).uv(:,2);
		uv_curr(:,j) = obj.features(i).uv(:,1);
		i_inliered(j) = i;
		j = j + 1;
	end
end
x_prev = K \ [uv_prev; ones(1, nFeature2DInliered)];
x_curr = K \ [uv_curr; ones(1, nFeature2DInliered)];

% Find reasonable rotation and translational vector
max_num = 0;
for i = 1:length(R_vec)
	R1 = R_vec{i};
	t1 = t_vec{i};
	
	m_ = cell(nFeature2DInliered, 1);
	t_ = zeros(3*nFeature2DInliered, 1);
	for j = 1:nFeature2DInliered
		m_{j} = skew(x_curr(:,j))*R1*x_prev(:,j);
		t_(3*(j-1)+1:3*j) = skew(x_curr(:,j))*t1;
	end
	M = blkdiag(m_{:});
	M(:, nFeature2DInliered+1) = t_;
	
	[~,~,V] = svd(M.'*M);
	lambda_prev = V(1:end-1,end).'/(V(end,end));
	
	lambdax_curr = bsxfun(@plus, bsxfun(@times, lambda_prev, R1*x_prev), t1);
	lambda_curr = lambdax_curr(3,:);
	
	inlier = find(lambda_prev > 0 & lambda_curr > 0);
	
	if length(inlier) > max_num
		max_num = length(inlier);
		max_inlier = inlier;
		point_prev = [bsxfun(@times, lambda_prev(inlier), x_prev(:,inlier));ones(1,length(inlier))];
		
		R = R1;
		t = t1;
	end
end

%% STORE
% Store 3D characteristics in features
if max_num < nFeature2DInliered*0.5
	R = [];
	t = [];
	
else
	idx = i_inliered(max_inlier);
	for i = 1:length(idx)
		% Update 3D point of features with respect to the current image
		% frame
		obj.features(idx(i)).point = point_prev(:,i);
		obj.features(idx(i)).is_3D_reconstructed = true;
	end
	obj.nFeature3DReconstructed = length(idx);
	
end