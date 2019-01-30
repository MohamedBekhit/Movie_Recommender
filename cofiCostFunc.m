function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));


% for i = 1:num_movies
%     for j = 1:num_users
%         if R(i, j) == 1
%             J = J + (X(i, :) * Theta(j, :)' - Y(i, j))^2 ;
%         end
%     end
% end

J = J + sum(sum(((X * Theta' - Y).^2).*R)); %vectorized

J = J/2 ;

%Regulization
J = J + lambda / 2 * (sum(sum((Theta).^2)) + sum(sum((X).^2)));

% for i = 1:num_movies
%     idx = find(R(i, :) == 1);
%     theta_temp = Theta(idx, :);
%     y_temp = Y(i, idx);
%     
%     X_grad(i, :) = (X(i, :) * theta_temp' - y_temp) * theta_temp;
% end
% 
% for j = 1:num_users
%     idx = find(R(:, j) == 1);
%     x_temp = X(idx, :);
%     y_temp = Y(idx, j);
%     
%     Theta_grad(j, :) = (x_temp * Theta(j, :)' - y_temp) * x_temp;
% end

%Regulized grad
X_grad = (R .* (X*Theta' - Y)) * Theta + lambda * X;
Theta_grad = (R .* (X*Theta' - Y))' * X + lambda * Theta;

grad = [X_grad(:); Theta_grad(:)];

end
