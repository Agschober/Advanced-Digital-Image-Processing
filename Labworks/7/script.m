% Progrmming assignment for AP3132-Advanced Digital Image Processing course
% Instructor: B. Rieger, F. Vos 
% Tutor: H. Heydarian
% Term: Q3-2021
%
% labwork  #7
%
clear
close all
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Problem #1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Bilateral filtering

% read the original input image 
in = readim('truinoise.tif');
dipshow(in);
f = gcf;
f.Name = 'Original image';
f.NumberTitle = 'off';

% convert dipimage to matrix
in = double(im2mat(in));

% bilateral filtering parameters
tonalSigma = 10;%TODO change
spatialSigma = 1;%TODO

% apply bilateral filter to the input image
out = bilateral(in, tonalSigma, spatialSigma);
dipshow(mat2im(out),'lin')
f = gcf;
f.Name = 'Filtered image';
f.NumberTitle = 'off';

out_gauss = gaussf(in,spatialSigma)
mean(out_gauss)
mean(mean(out))


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Problem #2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Deconvolution

% read the original input image
in = readim('trui.tif');
sz = size(in);
dipshow(in);
f = gcf;
f.Name = 'Original image';
f.NumberTitle = 'off';

%% TODO 2 - degrade input image with motion blur
% motion blurring kernel (H) in frequency domain
x = linspace(-3,3,sz(1));
y1 = sinc(4*x);
y2 = sinc(x);
H = y1'*y2;

% rotate by alpha degree
%H = mat2im(imrotate(H,45,'crop')); %needs image processing toolbox
H = cut(rotation(H,-45/180*pi,'linear'),size(H));

% compute the Fourier transform of the original image
IF = ft(in);
% apply the blurring kernel in frequency domain
IFd = IF.*H;
% take the inverse fourier transform
Iblur = real(ift(IFd));

% plot the blurred image
f=dipshow(Iblur,'lin');
f.Name = 'Blurred image';
f.NumberTitle = 'off';

%% TODO 2 - apply inverse filtering 

f.Name = 'Deblurred image';
f.NumberTitle = 'off';


%%
% (b) Wiener filter

% start again here with the blurred image

% TODO 3
% add noise to the blurred image
J = noise(Iblur, 'gaussian', 0.1, 0);

% display the degraded image
dipshow(J);
f = gcf;
f.Name = 'Noisy image';
f.NumberTitle = 'off';


% apply Wiener filter to corrupted image with proper K parameter
% apply the inverse blurring kernel in frequency domain

% display the recovered image
dipshow(Jr,'lin')
f = gcf;
f.Name = 'Wiener image';
f.NumberTitle = 'off';
