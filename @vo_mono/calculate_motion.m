function flag = calculate_motion( obj )
%% CALCULATION
if obj.step == 1
	flag = true;
	return;
end

% Calculate ego-motion
E = obj.essential;

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

% Verity 4 solutions
[R, t] = obj.verify_solutions(R_vec, t_vec);
[R, t] = obj.scale_propagation(R, t);

%% STORE
if obj.nFeature3DReconstructed < obj.params.thInlier
	warning('there are a few 3D POINT INLIERS');
	flag = false;
else
	% Save solution
	step = obj.step;
	
	obj.TRec{step} = [R' -R'*t; 0 0 0 1];
	obj.TocRec{step} = obj.TocRec{step-1} * obj.TRec{step};
	obj.PocRec(:,step) = obj.TocRec{step} * [0 0 0 1]';

	flag = true;
end
