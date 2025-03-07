% Progrmming assignment for AP3132-Advanced Digital Image Processing course
% Instructor: B. Rieger, F. Vos 
% Tutor: H. Heydarian
% Term: Q3-2021
%
% Labwork #4
%
clear all, close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Problem #1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Harris corner detection

% generate a right-angles triangle with angles alpha and 90-alpha
alpha = pi/10;
I = triangle_local(alpha);
dipshow(I)
%%
% parameter settings
sigma_grad = 1;
sigma_tensor = 3;%note this must be larger than sigma_grad
k = 0.04;
thresh = 10;

% compute the cornerness R with proper parameters
[xy, R] = harris(I, sigma_grad, sigma_tensor, k, thresh);
dipshow(R)
colormap('jet')

%%
% compute cornerness for trui.tif and water.tif
% TODO2
J = readim('water.tif');
dipshow(J)

[xy, R] = harris(J, 0.5, 1.5, k, thresh);
dipshow(R)

[xy, R] = harris(J, 1.0, 3.0, k, thresh);
dipshow(R)

pause

J = readim('trui.tif');
dipshow(J)

[xy, R] = harris(J, 0.5, 1.5, k, thresh);
dipshow(R)

[xy, R] = harris(J, 1.0, 3.0, k, thresh);
dipshow(R)

pause

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Problem #2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% PCA

clear all
close all
clc

% read the input images (6 different channels)
I = readim(['multispecData.ics']);
% this is a 3D image where the 6 different channels are stored in the thrid
% dimension
h=dipshow(I);
h.Name='6 input channels';
h.NumberTitle='off';
% use the tab 'Action' step through slices to see each channel or use the
% keyboard shortcuts 'n' (next) and 'p' (previous) 

% get dimensions
In = im2mat(I); % convert to Maltab 3D array for processing
[M, N, channel] = size(In);

% reshape the six images to a matrix X of size (height*width)x6
% i.e. each image become a vector of size height*width 
% TODO 3
X = reshape(In, M*N, channel);

%chceck whether we can construct a single spectral image from the reshaped
%matrix
%Answer: YES :)
%Y = reshape(X(:,1), M,N);
%dipshow(Y)

% compute the mean (mx) and covariance (Cx) of X
% TODO 4
mx = mean(X);
Cx = cov(X);


% compute the eigenvalues and eigenvectors of Cx.
% check the eigenvalues of Cx with the table in the manual
% TODO 5
[eigenvect, eigenvals] = eig(Cx);

% construct the matrix U whose rows are 
% formed from the eigenvectors U1 of Cx arranged in descending value of their
% eigenvalues
[eigenvals, sorting]  = sort(unique(eigenvals(eigenvals ~= 0)), 'descend')
eigenvals = diag(eigenvals);

% compute the principle components (Hotelling transform)
U = eigenvect(sorting, :);

% plot the principle components
PC = (X - mx)*U;
dipshow(reshape(PC, M, N, channel))

pause

% reconstruction using the main k components
% TODO 6
components = 5;
Uk = zeros(size(U));
Uk(1:components,:) = U(1:components,:);
reconstruction = PC*transpose(Uk) + mx;

dipshow(reshape(reconstruction, M, N, channel))

% form the matrix Uk from the k eigenvectors corresponding to the k
% largest eigenvalues
% TODO 7

% reconstruction using the k components
% projection of 6 channels onto 2 eigenvectors
% TODO 8

%AS: how are 7 and 8 different from 6?

% convert xp vectors to MxN images Inew

% compute the difference image
error = X - reconstruction;
dipshow(reshape(error, M, N, channel));

% compute the mean squared error between x and xp using the formula in the
% lecture notes
computed_error = mean(error.^2, 'all')
estimated_error = sum(eigenvals(components + 1:end, components + 1:end), 'all')