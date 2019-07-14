function [quatVector] = convert_r2q(rotMtxBody)
% Project:   Patch-based illumination-variant DVO
% Function: r2q
%
% Description:
%   This function convert rotation matrix to unit orientation quaternion
%
% Example:
%   OUTPUT:
%   quatVector: quaternion vector composed of [qw qx qy qz]
%
%   INPUT:
%   rotMtx = Rotation Matrix [3x3]
%               defined as [Inertial frame] = rotMtxBody * [Body frame]
%
% NOTE:
%
% Author: Pyojin Kim
% Email: pjinkim1215@gmail.com
% Website:
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% log:
% 2015-02-06: Complete
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%

q1 = 0.5 * sqrt(1+rotMtxBody(1,1)+rotMtxBody(2,2)+rotMtxBody(3,3));
q2 = (1/(4*q1))*(rotMtxBody(3,2)-rotMtxBody(2,3));
q3 = (1/(4*q1))*(rotMtxBody(1,3)-rotMtxBody(3,1));
q4 = (1/(4*q1))*(rotMtxBody(2,1)-rotMtxBody(1,2));

quatVector = [q1;q2;q3;q4];

end

