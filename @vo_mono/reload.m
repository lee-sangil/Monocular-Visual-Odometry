function obj = reload( obj )

obj.features = obj.features_backup;
obj.nFeature = length(obj.features);