%
% Copyright (c) 2015, Yarpiz (www.yarpiz.com)
% All rights reserved. Please read the "license.txt" for license terms.
%
% Project Code: YPML110
% Project Title: Implementation of DBSCAN Clustering in MATLAB
% Publisher: Yarpiz (www.yarpiz.com)
% 
% Developer: S. Mostapha Kalami Heris (Member of Yarpiz Team)
% 
% Contact Info: sm.kalami@gmail.com, info@yarpiz.com
%

function [IDX, Cen, bin, isnoise] = DBSCAN(X, epsilon, MinPts)

    C=0;
    
    n=size(X,1);
    IDX=zeros(n,1);
    
    D=pdist2(X,X);
    
    visited=false(n,1);
    isnoise=false(n,1);
    
	for i=1:n
		if ~visited(i)
			visited(i)=true;
			
			Neighbors=RegionQuery(i);
			if numel(Neighbors)<MinPts
				% X(i,:) is NOISE
				isnoise(i)=true;
			else
				C=C+1;
				ExpandCluster(i,Neighbors,C);
			end
			
		end
		
	end
	
	bin = histcounts(IDX, (0:max(1,max(IDX)))+0.5);
    
    % Sort by the number of group
    [~, sIdx] = sort(bin, 'descend');
    
    % Re-arrange
    IDX_sort = zeros(size(IDX));
    for i = 1:max(IDX)
        IDX_sort(IDX==sIdx(i)) = i;
    end
    
    IDX = IDX_sort;
	
	Cen = zeros(size(X,2), max(IDX));
	for i = 1:max(IDX)
		Cen(:,i) = mean(X(IDX==i,:),1).';
	end
	
	bin = histcounts(IDX, (0:max(1,max(IDX)))+0.5);
    	
    function ExpandCluster(i,Neighbors,C)
        IDX(i)=C;
        
        k = 1;
        while true
            j = Neighbors(k);
            
            if ~visited(j)
                visited(j)=true;
                Neighbors2=RegionQuery(j);
                if numel(Neighbors2)>=MinPts
                    Neighbors=[Neighbors Neighbors2];   %#ok
                end
            end
            if IDX(j)==0
                IDX(j)=C;
            end
            
            k = k + 1;
            if k > numel(Neighbors)
                break;
            end
        end
    end
    
    function Neighbors=RegionQuery(i)
        Neighbors=find(D(i,:)<=epsilon);
    end

end



