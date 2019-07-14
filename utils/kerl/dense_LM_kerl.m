function xi = dense_LM_kerl(c1, c2, d1, d2, K1, K2)

mode = 2;

% initialization
xi = [0 0 0 0 0 0]';

% pyramid levels
for lvl = 5:-1:1
    % get downscaled image, depth image, and K-matrix of down-scaled image.
    [IRef, K1lvl] = downscaleImage(c1,K1,lvl);
    I = downscaleImage(c2,K2,lvl);
    [DRef] = downscaleDepth(d1,lvl);
	lambda = 0.01;
	
    % just do at most 20 steps.
    errLast = 1e10;
    for i = 1:50
		
		switch mode
			
			case 1
				% calculate Jacobian of residual function (Matrix of dim (width*height) x 6)
				%[Jac, residual] = deriveErrNumeric(IRef,DRef,I,xi,Klvl);   % ENABLE ME FOR NUMERIC DERIVATIVES
				[Jac, residual] = deriveErrAnalytic(IRef,DRef,I,xi,K1lvl);   % ENABLE ME FOR ANALYTIC DERIVATIVES
				
				% just take the pixels that have no NaN (e.g. because
				% out-of-bounds, or because the didnt have valid depth).
				valid = ~isnan(sum(Jac,2));
				residualTrim = residual(valid,:);
				JacTrim = Jac(valid,:);
				
				% compute Huber Weights
				huber = ones(size(residual));
				huberDelta = 4/255;
				huber(abs(residual) > huberDelta) = huberDelta ./ abs(residual(abs(residual) > huberDelta));
				huberTrim = huber(valid);
				
				% do LM
				H = (JacTrim' * (repmat(huberTrim,1,6) .* JacTrim));
				upd = - (H + lambda * diag(diag(H)))^-1 * JacTrim' * (huberTrim .* residualTrim);
				
				% MULTIPLY increment from left onto the current estimate.
				lastXi = xi;
				xi = se3Log(se3Exp(upd) * se3Exp(xi));
% 				xi = upd + xi;
				
				% get mean and display
				err = mean((huberTrim.*residualTrim) .* residualTrim);
				
				% break if no improvement
				if(err >= errLast)
					lambda = lambda * 2;
					xi = lastXi;
					
					if(lambda > 2)
						break;
					end
				else
					lambda = lambda /1.5;
				end
				errLast = err;
		
			case 2
				% calculate Jacobian of residual function (Matrix of dim (width*height) x 6)
				%[Jac, residual] = deriveErrNumeric(IRef,DRef,I,xi,Klvl);   % ENABLE ME FOR NUMERIC DERIVATIVES
				[Jac, residual] = deriveErrAnalytic(IRef,DRef,I,xi,K1lvl);   % ENABLE ME FOR ANALYTIC DERIVATIVES
				
				% just take the pixels that have no NaN (e.g. because
				% out-of-bounds, or because the didnt have valid depth).
				valid = ~isnan(sum(Jac,2));
				residualTrim = residual(valid,:);
				JacTrim = Jac(valid,:);
				
				% do Gauss-Newton step
				upd = - (JacTrim' * JacTrim)^-1 * JacTrim' * residualTrim;
				
				% MULTIPLY increment from left onto the current estimate.
				% 		xi = xi + upd;
				xi = se3Log(se3Exp(upd) * se3Exp(xi));
% 				xi = upd + xi;
				
				% get mean and display
				err = mean(residualTrim .* residualTrim);
				
				
				%calcErr(c1,d1,c2,xi,K);
				
				% break if no improvement
				if(err / errLast > 0.999)
					break;
				end
				errLast = err;
		end
    end
    
end