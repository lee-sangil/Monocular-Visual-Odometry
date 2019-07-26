function flag = calculate_motion( obj )
%% CALCULATION
if obj.step == 1
	flag = true;
	return;
end

R_vec = obj.R_vec;
t_vec = obj.t_vec;

[R, t, success, inlier, outlier] = obj.findPoseFrom3DPoints();
if success == false
	% Verity 4 solutions
	[R_, t_, success] = obj.verify_solutions(R_vec, t_vec);
	
	if success == false
		flag = false;
		return;
	end
	
	% Update 3D points
	[R, t, success, inlier, outlier] = obj.scale_propagation(R_, t_);
	
	if success == false
		flag = false;
		return;
	end
	
	[T, Toc, Poc] = obj.update3DPoints(R, t, inlier, outlier, 'w/oPnP');
else
	% Update 3D points
	[T, Toc, Poc] = obj.update3DPoints(R, t, inlier, outlier, 'w/PnP');
end
obj.scale_initialized = true;

%% Store
if obj.nFeature3DReconstructed < obj.params.thInlier
	warning('there are a few 3D POINT INLIERS');
	flag = false;
else
	% Save solution
	step = obj.step;
	
	obj.TRec{step} = T;
	obj.TocRec{step} = Toc;
	obj.PocRec(:,step) = Poc;
	
	flag = true;
end