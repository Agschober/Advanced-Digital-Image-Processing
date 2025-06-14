function [indices, newMean, stop_flag] = MoveMean(data, oldMean, radius, stopThreshold)

% get the number of data points (data is dimxN where dim is the
% dimension)
[~,N] = size(data);

% TODO 5
% compute the sum of the squared differences of the mean
% (oldMean) to all data points (data)
distances = sum((data - oldMean).^2, 1);

% TODO 6
% find all the data points within the neighbourhood defined by the
% oldMean and the radius
indices      = find(distances < radius);

% TODO 7
% compute the mean (newMean) of the points within that neighborhood
newMean      = mean(data(:, indices), 2);

% TODO 8
% set stop_flag to 1 if the Euclidean distance between the newMean and oldMean
% is below stopThreshold. Look at norm() funcion in MATLAB docs.
stop_flag = (norm(newMean - oldMean) < stopThreshold);

end
