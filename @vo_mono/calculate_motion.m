function flag = calculate_motion( obj )
%% CALCULATION
if obj.step == 1
	flag = true;
	return;
end

R_vec = obj.R_vec;
t_vec = obj.t_vec;

[R, t, success1] = obj.findPoseFrom3DPoints();
if success1 == false
	% Verity 4 solutions
	[R_, t_, success2] = obj.verify_solutions(R_vec, t_vec);
	
	if success2 == false
		warning('There are no meaningful R, t.');
		flag = false;
		return;
	end
	
	% Update 3D points
	[R, t, success3, inlier, outlier] = obj.scale_propagation(R_, t_);
	
	if success3 == false
		warning('There are a few inliers matching scale.');
		flag = false;
		return;
	end
	
	[T, Toc, Poc] = obj.update3DPoints(R, t, inlier, outlier, 'w/oPnP');
else
	[R_, t_, ~] = obj.verify_solutions(R_vec, t_vec);
	
	% Update 3D points
	[R_, t_, success3, inlier, outlier] = obj.scale_propagation(R_, t_);
	
	% Update 3D points
	[T, Toc, Poc] = obj.update3DPoints(R, t, inlier, outlier, 'w/PnP', R_, t_, success3);
end
obj.scale_initialized = true;

%% Store
if obj.nFeature3DReconstructed < obj.params.thInlier
	warning('there are a few inliers reconstructed in 3d.');
	flag = false;
else
	% Save solution
	step = obj.step;
	
	obj.TRec{step} = T;
	obj.TocRec{step} = Toc;
	obj.PocRec(:,step) = Poc;
	
	flag = true;
end

if ~isempty(obj.pose)
	if norm(obj.pose(:,obj.step)-obj.PocRec(:,obj.step)) > 3
		a = 1;
	end
end