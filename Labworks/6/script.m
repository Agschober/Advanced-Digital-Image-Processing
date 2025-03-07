% Progrmming assignment for AP3132-Advanced Digital Image Processing course
% Instructor: B. Rieger, F. Vos 
% Tutor: H. Heydarian
% Term: Q3-2020
%
% Labwork Wavelet
%
clear all
close all
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Problem #1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1D wavelet

% input 1D signal
f = [1 4 -3 0]



% apply 1D haar transform, complete code in haar_wavelet1D.m
wf = haar_wavelet1D(f)

% apply inverse 1D haar transform, complete code in inv_haar_wavelet1D.m
iwf = inv_haar_wavelet1D(wf)
%% denoising using haar wavelet

% input sine signal
x = linspace(-3.14,3.14,256);
y = sin(x);

% display the input signal
figure;
subplot(1,2,1); plot(x,y);
title('input signal')

% add additve white gaussian noise to signal
z = noise(y, 'gaussian', 0.25, 0);
z = im2mat(z);

% display the noisy signal
subplot(1,2,2); plot(x,z);
title('noisy signal')
%%
% apply 1D haar transform
zw = haar_wavelet1D(z);


% TODO 3
% modify the wavelet coeffiecient (zw) for denoising
zw1=zw;

% apply inverse wavelet transform
zr = inv_haar_wavelet1D(zw1);

% display the result
figure;plot(x,zr);
title('recovered signal')
%% (OPTIONAL) YOU NEED MATLAB wavelet toolbox TO RUN THIS SECTION

% compare your result with matlab function wdenoise
[XDEN,ii] = wdenoise(z,8,'Wavelet','haar');

% display the matlab result
figure;plot(x, XDEN);
title('recovered signal using matlab')


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Problem #2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 2D discrete Haar wavelet transform

% read the original input image
image = double(imread('truinoise.tif'));
dipshow(mat2im(image))
f = gcf;
f.Name = 'Original image';
f.NumberTitle = 'off';

% create a copy of the original image
w_image = image;

% set the wavelet type to haar
wavename = 'haar';

% TODO 4
% use dwt2() to implement full 2D wavelet decomposition
for j=7:-1:0
    
    % TODO4a - select the coarse scale (subimage)
    
    
    % apply dwt2 to coarse scale
   
    
    % TODO4b - build-up the wavelet component according to Figure 1 in labwork
    % cA->T_phi, cH->T^H_psi, cV->T^V_psi, cD->T^D_psi
    tmp = [cA,cH; cV,cD];
    
    % TODO4c - substitute wavelet component into coarse scale
  
end

% display the result
dipshow(mat2im(w_image));
f = gcf;
f.Name = 'Wavelet decomposition';
f.NumberTitle = 'off';
%%

% create a copy of the wavelet decomposed image
iw_image = w_image;

% TODO 5
% use idwt2() to implement full 2D wavelet reconstruction
for j = 0:7
    
    % TODO5a - select the coarse scale (subimage)
 
    
    % TODO5b - set the coarse scale size

    
    % TODO5c - apply idwt2 to coarse scale
    % carefully choose cA and details matrices cH, cV, and cD
    tmp = idwt2(cA, cH, cV, cD, wavename);
    
    % TODO4d - substitute wavelet component into coarse scale
    
end

% display the result
dipshow(mat2im(iw_image))
f = gcf;
f.Name = 'Reconstruction';
f.NumberTitle = 'off';
%%

% TODO 6
% Analysis of wavelet coefficients

% create a copy of the wavelet decomposed image
denoise_image = w_image;
%TODO discard part of the image as you think is reasonable


for j = 0:7
    a = denoise_image(1:2^(j+1),1:2^(j+1));    
    m = 2^(j+1);
    tmp = idwt2(a(1:m/2,1:m/2), a(1:m/2,m/2+1:end), a(m/2+1:end,1:m/2), a(m/2+1:end,m/2+1:end), 'haar');
    denoise_image(1:2^(j+1),1:2^(j+1)) = tmp;
end

dipshow(mat2im(denoise_image))
f = gcf;
f.Name = 'Denoised image';
f.NumberTitle = 'off';
