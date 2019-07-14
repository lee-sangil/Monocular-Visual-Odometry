function [RPE, data] = calculate_rpe(est, true, fs)
% est: true: tx ty tz qw qx qy qz: 7 dof

m = size(est,2);

RPE = 0;
for i = 1:m-fs
	Pd = convert(true(:,i+fs));
	P = convert(true(:,i));
	Qd = convert(est(:,i+fs));
	Q = convert(est(:,i));
	E = (Q\Qd)\(P\Pd);
	RPE = RPE + sum(E(1:3,4).^2);
	data(i,1) = sqrt(mean(E(1:3,4).^2));
end

RPE = (RPE / (m-fs))^(1/2);

end

function T = convert(q)

T(1:3,1:3) = convert_q2r(q(4:7));
T(1:3,4) = q(1:3);
T(4,4) = 1;

end