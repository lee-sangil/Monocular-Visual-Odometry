function [Chypotheses, Cerr] = hypothesis_clustering(hypotheses, X1, X2, funcVectorize, funcMatrix, funcClustering, funcEstimateHypothesis, funcEvaluateHypothesis)

nHypothesis = length(find(~cellfun(@isempty, hypotheses)));

% Construct motion model
motion = zeros(6, nHypothesis);
for iHypothesis = 1:nHypothesis
	motion(:,iHypothesis) = funcVectorize(hypotheses{iHypothesis});
end

% Clustering motion hypotheses
[~, C, ~] = funcClustering(motion);
nHypothesis = size(C, 2);
Cerr = zeros(1,size(X1,2));

% Refine transformation while expanding inliers
jHypothesis = 0;
for iHypothesis = 1:nHypothesis
	H = funcMatrix(C(:,iHypothesis));
	
	max_inlier = 0;
	while true
		
		[err, inlier] = funcEvaluateHypothesis(H, X1, X2);
		
		if sum(inlier) > max_inlier
			max_inlier = sum(inlier);
			H = funcEstimateHypothesis(X1(:,inlier), X2(:,inlier));
		else
			break;
		end
	end
	
	if max_inlier > 10
		jHypothesis = jHypothesis + 1;
		hypotheses{jHypothesis} = H;
		Cerr(jHypothesis,:) = err;
	end
end

nHypothesis = jHypothesis;
Chypotheses = hypotheses(1:nHypothesis);
Cerr = Cerr(1:nHypothesis,:);