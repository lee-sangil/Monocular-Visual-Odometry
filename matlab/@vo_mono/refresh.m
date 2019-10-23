function obj = refresh( obj )

obj.nFeatureMatched = 0;
obj.nFeature2DInliered = 0;
obj.nFeature3DReconstructed = 0;
obj.nFeatureInlier = 0;

for i = 1:obj.nFeature
	obj.features(i).is_matched = false;
	obj.features(i).is_2D_inliered = false;
	obj.features(i).is_3D_reconstructed = false;
end