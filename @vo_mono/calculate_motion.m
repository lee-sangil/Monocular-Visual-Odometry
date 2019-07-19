function flag = calculate_motion( obj )

if obj.step == 1
	flag = true;
	return;
end

%% calculate ego-motion
E = obj.essential;

[U, ~, V] = svd(E);
if det(U) <  0
	U(:,3) = -U(:,3);
end
if det(V) < 0
	V(:,3) = -V(:,3);
end

%% extract rotational and translational movement
W = [0 -1 0; 1 0 0; 0 0 1];
R_vec{1} = U*W*V.';
R_vec{2} = U*W*V.';
R_vec{3} = U*W.'*V.';
R_vec{4} = U*W.'*V.';
t_vec{1} = U(:,3);
t_vec{2} = -U(:,3);
t_vec{3} = U(:,3);
t_vec{4} = -U(:,3);

%% verity 4 solutions
[R, t] = obj.verify_solutions(R_vec, t_vec);
[R, t] = obj.scale_propagation(R, t);

if obj.nFeature3DReconstructed < obj.params.thInlier
	warning('there are a few 3D POINT INLIERS');
	flag = false;
else
	%% save solution
	step = obj.step;
	
	obj.TRec{step} = [R t; 0 0 0 1];
	obj.TocRec{step} = obj.TRec{step} * obj.TocRec{step-1};
	obj.PocRec(:,step) = obj.TocRec{step} \ [0 0 0 1]';
	
	[idx, ~] = seek_index(obj, obj.nFeature, [obj.features(:).is_3D_init]);
	for i = idx
% 		obj.features(i).point = norm(t)*obj.features(i).point;
		obj.features(i).transform_from_init = obj.features(i).transform_from_init * obj.TRec{step};
		obj.features(i).transform_to_step = inv(obj.features(i).transform_from_init);
	end

	flag = true;
end
