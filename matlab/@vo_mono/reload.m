function obj = reload( obj )

obj.features = obj.features_backup;
obj.nFeature = length(obj.features);

obj.TRec{obj.step} = eye(4);
obj.TocRec{obj.step} = obj.TocRec{obj.step};
obj.PocRec(:,obj.step) = obj.PocRec(:,obj.step-1);