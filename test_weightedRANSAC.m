function test_weightedRANSAC

x = 0:0.1:10;
z(1:50) = 2*x(1:50) + 0.2*randn(size(x(1:50)));
z(51:101) = 1.5*x(50)+0.5*x(51:101) + 0.2*randn(size(x(51:101)));

figure(1);
plot(x,z);hold on;

ransacCoef.iterMax = 1e3;
ransacCoef.minPtNum = 2;
ransacCoef.thInlrRatio = 0.99;
ransacCoef.thDist = 0.2;
ransacCoef.thDistOut = 0.2;
ransacCoef.funcFindF = @(x,y) findLine(x,y);
ransacCoef.funcDist = @(f,x,y) abs(f(1)*x+f(2)-y);
ransacCoef.weight = ones(size(x));
% ransacCoef.weight = x;

for i = 1:100
	f = ransac(x,z, ransacCoef);
	y = f(1)*x+f(2);
	plot(x,y,'r');
end

end

function f = findLine(x,y)
	x_hat = [x' ones(length(x),1)];
	f = (y*x_hat)/(x_hat.'*x_hat);
% 	f(1) = (y(2)-y(1)) / (x(2)-x(1));
% 	f(2) = (x(2)*y(1)-y(2)*x(1)) / (x(2)-x(1));
end