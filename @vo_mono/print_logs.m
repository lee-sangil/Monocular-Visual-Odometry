function obj = print_logs( obj, pkg, timePassed )

% fprintf('step: %03d | #inliers: %3d | scale: %5.4f | time passed: %3.2f ms\n', ...
% 			pkg.step, obj.get_nFeatureInlier, norm(obj.TRec{obj.step-1}(1:3,4)), timePassed*1000);