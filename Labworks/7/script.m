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

%{
 tonalSigma = 5;%TODO change
spatialSigma = 5;%TODO

% apply bilateral filter to the input image
out = bilateral(in, tonalSigma, spatialSigma);
dipshow(mat2im(out),'lin')
f = gcf;
f.Name = 'Filtered image';
f.NumberTitle = 'off';

mean(mean(out))

% bilateral filtering parameters
tonalSigma = 10;%TODO change
spatialSigma = 5;%TODO

% apply bilateral filter to the input image
out = bilateral(in, tonalSigma, spatialSigma);
dipshow(mat2im(out),'lin')
f = gcf;
f.Name = 'Filtered image';
f.NumberTitle = 'off';

mean(mean(out))

% bilateral filtering parameters
tonalSigma = 20;%TODO change
spatialSigma = 5;%TODO

% apply bilateral filter to the input image
out = bilateral(in, tonalSigma, spatialSigma);
dipshow(mat2im(out),'lin')
f = gcf;
f.Name = 'Filtered image';
f.NumberTitle = 'off';

mean(mean(out)) 
%}


% bilateral filtering parameters
tonalSigma = 200;%TODO change
spatialSigma = 5;%TODO

% apply bilateral filter t
% 
% o the input image
out = bilateral(in, tonalSigma, spatialSigma);
dipshow(mat2im(out),'lin')
f = gcf;
f.Name = 'Filtered image';
f.NumberTitle = 'off';

out_gauss = gaussf(in,spatialSigma)
mean(out_gauss)
mean(mean(out))

pause
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
dipshow(Iblur,'lin');
%f.Name = 'Blurred image';
%f.NumberTitle = 'off';

%% TODO 2 - apply inverse filtering 
G = ft(Iblur); 

%typical inverse convolution doesn't work since the sinc function has zeros
%But since we everything about the convolution process we know that if we find the PSF of the inverse operation (an image blur in the opposite direction)
%the image can be fully reconstructed.

F_approx = zeros(sz);
for i = 1:255
    for j = 1:255
        if abs(H(j,i)) < 1E-6
            F_approx(j,i) = 0;
        else
            F_approx(j,i) = IFd(j,i)/H(j,i);
        end
    end
end

recon = abs(ift(transpose(F_approx)));

dipshow(recon);
%f =gcf;
%f.Name = 'Deblurred image';
%f.NumberTitle = 'off';

pause
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
K = 0.001;
G = ft(J);
Jr = conj(H)./(abs(H).^2 + K).*G;

%G = ft(image) + ft(noise), the noise is a constant, the image has some structure to it
% (ft(image) + ft(noise))H*/(|H|^2 + K) shows that it is a fine balance between suppressing noise and reconstructing the image

Jr = abs(ift(Jr));

% display the recovered image
f = dipshow(Jr,'lin')
f = gcf;
f.Name = 'Wiener image';
f.NumberTitle = 'off';
