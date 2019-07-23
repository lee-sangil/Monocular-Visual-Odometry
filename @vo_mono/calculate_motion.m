function flag = calculate_motion( obj )
%% CALCULATION
if obj.step == 1
	flag = true;
	return;
end

R_vec = obj.R_vec;
t_vec = obj.t_vec;

% Verity 4 solutions
[R_, t_] = obj.verify_solutions(R_vec, t_vec);

% [R, t] = obj.findPoseFrom3DPoints();
[R, t] = obj.scale_propagation(R_, t_);

if t(3) > 0
	a = 1;
end

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
