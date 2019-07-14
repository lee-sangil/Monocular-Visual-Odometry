function [f, inlierIdx, outlierIdx] = ransac( x, y, ransacCoef )
%[f inlierIdx] = ransac1( x,y,ransacCoef,funcFindF,funcDist )
%	Use Adaptive RANdom SAmple Consensus to find a fit from X to Y.
%	X is M*n matrix including n points with dim M, Y is N*n;
%	The fit, f, and the indices of inliers, are returned.
%
%	RANSACCOEF is a struct with following fields:
%	minPtNum,iterNum,thDist,thInlrRatio
%	MINPTNUM is the minimum number of points with whom can we 
%	find a fit. For line fitting, it's 2. For homography, it's 4.
%	ITERNUM is the number of iteration, THDIST is the inlier 
%	distance threshold and ROUND(THINLRRATIO*n) is the inlier number threshold.
%
%	FUNCFINDF is a func handle, f1 = funcFindF(x1,y1)
%	x1 is M*n1 and y1 is N*n1, n1 >= ransacCoef.minPtNum
%	f1 can be of any type.
%	FUNCDIST is a func handle, d = funcDist(f,x1,y1)
%	It uses f returned by FUNCFINDF, and return the distance
%	between f and the points, d is 1*n1.
%	For line fitting, it should calculate the dist between the line and the
%	points [x1;y1]; for homography, it should project x1 to y2 then
%	calculate the dist between y1 and y2.

iterMax = ransacCoef.iterMax;
minPtNum = ransacCoef.minPtNum;
thInlrRatio = ransacCoef.thInlrRatio;
thDist = ransacCoef.thDist;
thDistOut = ransacCoef.thDistOut;
funcFindF = ransacCoef.funcFindF;
funcDist_ = ransacCoef.funcDist;
if isfield(ransacCoef, 'weight')
	w = ransacCoef.weight;
	funcDist = @(f,x,y)funcDist_(f,x,y,w);
else
	funcDist = @(f,x,y)funcDist_(f,x,y);
end

ptNum = size(x,2);

iterNum = inf;
max_inlier = 0;
inlierIdx = [];

it = 1;
while it < min(iterMax, iterNum)
	% 1. fit using  random points
	sampleIdx = randperm(ptNum, minPtNum);
	f1 = funcFindF(x(:,sampleIdx), y(:,sampleIdx));
	
	if iscell(f1)
		for c = 1:length(f1)
			% 2-1. count the inliers
			fi = f1{c};
			dist1 = funcDist(fi,x,y);
			in1 = find(dist1 < thDist);
			
			% 3-1. save parameter
			if length(in1) > max_inlier
				max_inlier = length(in1);
				inlierIdx = in1;
				InlrRatio = length(inlierIdx) / ptNum + eps;
				iterNum = floor(log(1-thInlrRatio) / log(1-InlrRatio^minPtNum));
			end
		end
	else
		% 2-2. count the inliers
		dist1 = funcDist(f1,x,y);
		in1 = find(dist1 < thDist);
		
		% 3-2. save parameter
		if length(in1) > max_inlier
			max_inlier = length(in1);
			inlierIdx = in1;
			InlrRatio = length(inlierIdx) / ptNum + eps;
			iterNum = floor(log(1-thInlrRatio) / log(1-InlrRatio^minPtNum));
		end
	end
	
	it = it + 1;
end

if isempty(inlierIdx)
	f = [];
	inlierIdx = [];
	outlierIdx = [];

else
	% 4. choose the coef with the most inliers
	f1 = funcFindF(x(:,inlierIdx), y(:,inlierIdx));
	if iscell(f1)
		max_inlier = 0;
		for c = 1:length(f1)
			% 4-1. count the inliers
			fi = f1{c};
			dist1 = funcDist(fi,x,y);
			in1 = find(dist1 < thDist);
			
			% 4-2. save parameter
			if length(in1) > max_inlier
				max_inlier = length(in1);
				max_idx = c;
			end
		end
		f = f1{max_idx};
	else
		f = f1;
	end
	dist = funcDist(f, x, y);
	
	ptArr = 1:ptNum;
	inlierIdx = find(dist < thDist);
	outlierIdx = find(dist > thDistOut);
% 	outlierIdx = ptArr(~ismember(ptArr, inlierIdx));
	
end

end