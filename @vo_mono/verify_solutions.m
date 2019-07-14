function [R, t] = verify_solutions( obj, R_vec, t_vec )
%% VERIFICATION
% Load parameters from object
K = obj.params.K;
nFeature = obj.nFeature;
nFeature2DInliered = obj.nFeature2DInliered;

% Extract homogeneous 2D point which is inliered with essential constraint
i_inliered = zeros(1, nFeature2DInliered);
j = 1;
uv1 = zeros(2, nFeature2DInliered);
uv2 = zeros(2, nFeature2DInliered);
for i = 1:nFeature
	if obj.features(i).is_2D_inliered
		uv1(:,j) = obj.features(i).uv(:,2);
		uv2(:,j) = obj.features(i).uv(:,1);
		i_inliered(j) = i;
		j = j + 1;
	end
end
x1 = K \ [uv1; ones(1, nFeature2DInliered)];
x2 = K \ [uv2; ones(1, nFeature2DInliered)];

% Find reasonable rotation and translational vector
max_num = 0;
for i = 1:length(R_vec)
	R1 = R_vec{i};
	t1 = t_vec{i};
	
	m_ = cell(nFeature2DInliered, 1);
	t_ = zeros(3*nFeature2DInliered, 1);
	for j = 1:nFeature2DInliered
		m_{j} = skew(x2(:,j))*R1*x1(:,j);
		t_(3*(j-1)+1:3*j) = skew(x2(:,j))*t1;
	end
	M = blkdiag(m_{:});
	M(:, nFeature2DInliered+1) = t_;
	
	[~,~,V] = svd(M.'*M);
	lambda1 = V(1:end-1,end).'/(V(end,end));
	
	lambda2x2 = bsxfun(@plus, bsxfun(@times, lambda1, R1*x1), t1);
	lambda2 = lambda2x2(3,:);
	
	inlier = find(lambda1 > 0 & lambda2 > 0);
	
	if length(inlier) > max_num
		max_num = length(inlier);
		max_inlier = inlier;
		point_ = [bsxfun(@times, lambda1(inlier), x1(:,inlier));ones(1,length(inlier))];
		
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
		obj.features(idx(i)).point = point_(:,i);
		obj.features(idx(i)).is_3D_reconstructed = true;
	end
	obj.nFeature3DReconstructed = length(idx);
	
end