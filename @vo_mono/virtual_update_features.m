function flag = virtual_update_features( obj, features_, points_, id_ )

features = obj.features;

if obj.nFeature > 0
    % Track the points
	points = zeros(2,obj.nFeature);
	validity = zeros(1,obj.nFeature);
	for i = 1:obj.nFeature
		[valid, loc] = ismember(features(i).id, id_);
		if valid
			points(:,i) = features_(:,loc);
			validity(i) = 1;
		end
	end
	
    for i = 1:obj.nFeature
        if validity(i) && features(i).life > 0
			features(i).life = features(i).life + 1;
			features(i).uv(:,2:features(i).life) = features(i).uv;
			features(i).uv(:,1) = points(:,i);
			features(i).is_matched = true;
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
