function flag = update_features( obj )

features = obj.features;

if obj.nFeature > 0
    % Track the points
	[points, validity] = obj.klt_tracker();
	
    for i = 1:obj.nFeature
        if validity(i) && features(i).life > 0
			features(i).life = features(i).life + 1;
			features(i).uv(:,2:features(i).life) = features(i).uv;
			features(i).uv(:,1) = points(:,i);
			features(i).is_matched = true;
			
			if norm(features(i).uv(:,1) - features(i).uv(:,end)) > obj.params.min_px_dist
				features(i).is_wide = true;
			end
			
			obj.nFeatureMatched = obj.nFeatureMatched + 1;
		else
			features(i).life = 0;
        end
    end
    
	if obj.nFeatureMatched < obj.params.thInlier
		warning('there are a few FEATURE MATCHES');
		flag = false;
	else
		flag = true;
		obj.features = features;
	end
else
	flag = true;
end
