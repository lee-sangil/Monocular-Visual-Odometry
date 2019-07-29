figure;
n = size(x_prev,2);
scatter(x_prev(1,:), x_prev(2,:),[],1:n);hold on
scatter(x_curr(1,:), x_curr(2,:),[],1:n);
axis off
set(gca, 'YDir', 'reverse', 'color', [0 0 0]);
set(gcf, 'color', [0 0 0]);
xlim([-1.5 1.5]);
ylim([-.8 .8]);

figure;
n = size(x_prev,2);
scatter(X_prev(1,:)./X_prev(3,:), X_prev(2,:)./X_prev(3,:),[],1:n);hold on
scatter(X_curr(1,:)./X_curr(3,:), X_curr(2,:)./X_curr(3,:),[],1:n);
axis off
set(gca, 'YDir', 'reverse', 'color', [0 0 0]);
set(gcf, 'color', [0 0 0]);
xlim([-1.5 1.5]);
ylim([-.8 .8]);

figure;
n = length(inliers);
scatter(X_prev(1,inliers)./X_prev(3,inliers), X_prev(2,inliers)./X_prev(3,inliers),[],1:n);hold on
scatter(X_curr(1,inliers)./X_curr(3,inliers), X_curr(2,inliers)./X_curr(3,inliers),[],1:n);
axis off
set(gca, 'YDir', 'reverse', 'color', [0 0 0]);
set(gcf, 'color', [0 0 0]);
xlim([-1.5 1.5]);
ylim([-.8 .8]);