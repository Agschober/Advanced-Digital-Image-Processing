% code taken from [C. Liu et al. 'SIFT Flow: Dense Correspondence across Scenes and its Applications' IEEE trans. Pattern Analysis and Machine Intelligence]
% original code can be found here https://people.csail.mit.edu/celiu/ECCV2008/
% modified to recreate results from [Y. Liu et al. 'Multi-focus image fusion with dense SIFT' Information Fusion]
% for the course AP3132 Advanced Digital Image Processing at TU Delft in 2025 (assignment description can be found at https://qiweb.tudelft.nl/adip/projects/topic_07/)
% written by A. Schober & S. Verstraaten

% Step 0. Load and downsample the images
clear
close all

im1=imread('../duo_images/flower/flower-fg.jpg');
im2=imread('../duo_images/flower/flower-bg.jpg');

[nx, ny, ncol] = size(im1)

% AS: for some reason the pictures from my phone weren't the same size, makes comparison difficult, implemented a small fix to fix the size:)
scale = 3;
nx = round(nx/scale);
ny = round(ny/scale);

im1=imresize(imfilter(im1,fspecial('gaussian',7,1.),'same','replicate'),[nx, ny],'bicubic');
im2=imresize(imfilter(im2,fspecial('gaussian',7,1.),'same','replicate'),[nx, ny],'bicubic');

im1=im2double(im1);
im2=im2double(im2);

gs1 = color2grayscale(im1);
gs2 = color2grayscale(im2);

figure; imshow(im1);
figure; imshow(im2);

% Step 1. Compute the dense SIFT image

% patchsize is half of the window size for computing SIFT
% gridspacing is the sampling precision

patchsize=8;
gridspacing=1;

Sift1=dense_sift(im1,patchsize,gridspacing);
Sift2=dense_sift(im2,patchsize,gridspacing);

%AS: normalization removed from dense_sift function, So Sift is full scale vector Sift_norm is normalized vector
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
num_angles = 8;
num_bins = 4;
[nrows, ncols, ~] = size(Sift1);

% normalize SIFT descriptors
Sift1_norm = reshape(Sift1, [nrows*ncols num_angles*num_bins*num_bins]);
Sift1_norm= normalize_sift(Sift1_norm);
Sift1_norm = reshape(Sift1_norm, [nrows ncols num_angles*num_bins*num_bins]);

[nrows, ncols, cols] = size(Sift2);

% normalize SIFT descriptors
Sift2_norm = reshape(Sift2, [nrows*ncols num_angles*num_bins*num_bins]);
Sift2_norm = normalize_sift(Sift2_norm);
Sift2_norm = reshape(Sift2_norm, [nrows ncols num_angles*num_bins*num_bins]);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% visualize the SIFT image
%figure;imshow(showColorSIFT(Sift1_norm));title('SIFT image 1');
%figure;imshow(showColorSIFT(Sift2_norm));title('SIFT image 2');

% step 1 +  IMAGE REGISTRATION
% AS: SIFT flow works REALLY WELL (as far as I know) FOR IMAGE REGISTRATION IT SEEMS -> not part of the assignment but definitely worth mentioning
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Step 3. SIFT flow matching

% prepare the parameters
SIFTflowpara.alpha=2;
SIFTflowpara.d=40;
SIFTflowpara.gamma=0.005;
SIFTflowpara.nlevels=4;
SIFTflowpara.wsize=5;
SIFTflowpara.topwsize=20;
SIFTflowpara.nIterations=60;

tic;[vx,vy,energylist]=SIFTflowc2f(Sift1_norm,Sift2_norm,SIFTflowpara);toc 

% Step 4.  Visualize the matching results
Im1=im1(patchsize/2:end-patchsize/2+1,patchsize/2:end-patchsize/2+1,:);
Im2=im2(patchsize/2:end-patchsize/2+1,patchsize/2:end-patchsize/2+1,:);
 
warpI2=warpImage(Im2,vx,vy);
figure;imshow(Im1);title('Image 1');
figure;imshow(warpI2);title('Warped image 2');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%AS: register SIFT images
Im2 = warpImage(im2, vx, vy);
gs2 = warpImage(gs2, vx, vy);
Sift2 = warpImage(Sift2, vx, vy);
Sift2_norm = reshape(Sift2, [nrows*ncols num_angles*num_bins*num_bins]);
Sift2_norm = normalize_sift(Sift2_norm);
Sift2_norm = reshape(Sift2_norm, [nrows ncols num_angles*num_bins*num_bins]); 

%The registerd image is just slightly smaller -> rescale 
gs1 = gs1(patchsize/2:end-patchsize/2+1,patchsize/2:end-patchsize/2+1,:);

[nx, ny] = size(gs1);

%Obtain activity maps
A1 = sum(Sift1(patchsize/2:end-patchsize/2+1,patchsize/2:end-patchsize/2+1,:,:), 3);
A2 = sum(Sift2(patchsize/2:end-patchsize/2+1,patchsize/2:end-patchsize/2+1,:,:), 3);

%step 2: initial decision map
M1 = zeros(size(gs1));
M2 = M1;

 for i = patchsize:nx - patchsize + 1
    for j = patchsize:ny - patchsize + 1
        s1 = sum(sum(A1(i - patchsize + 1: i, j - patchsize + 1: j)));
        s2 = sum(sum(A2(i - patchsize + 1: i, j - patchsize + 1: j)));
        
        if s1 > s2
            M1(i - patchsize + 1: i, j - patchsize + 1: j) = M1(i - patchsize + 1: i, j - patchsize + 1: j) + 1;
        elseif s2 > s1
            M2(i - patchsize + 1: i, j - patchsize + 1: j) = M2(i - patchsize + 1: i, j - patchsize + 1: j) + 1;
        end
    end
end 

%rough decision maps
D1 = M2 == 0;
D2 = M1 == 0;

D_init = D1 + (1 - ((1-D2).*D1 + D2))*0.5;
figure;imshow(D_init)

%add post-processing step to D1 and D2
%remove small regions and holes in large areas (closing, opening or some other morphological operation)
n_filt = 1;
dim_filt_x = ceil(nx/10);
dim_filt_y = ceil(ny/10);
filt_type = 'rectangular';

tic
for i = 1:n_filt
    D1 = medif(D1, [dim_filt_x, dim_filt_y], filt_type);
    D2 = medif(D1, [dim_filt_x, dim_filt_y], filt_type);
end
toc

D1 = im2mat(D1);
D2 = im2mat(D2);

D_init = D1 + (1 - ((1-D2).*D1 + D2))*0.5;
dipshow(D_init)

%step 3: refine decision map using spatial frequency on image patch
D_fin = D_init;
cases = find(D_init == 0.5);

i_indices = 1:nx;
j_indices = 1:ny;
[j_indices, i_indices] = meshgrid(j_indices,i_indices);
i_indices = i_indices(cases);
j_indices = j_indices(cases);

tic
for k = 1:size(cases,1)
    i = i_indices(k);
    j = j_indices(k);

    if i - patchsize/2 < 0
        i = patchsize/2;

    elseif i + patchsize/2 > nx
        i = nx - patchsize/2;

    end

    if j - patchsize/2 < 0
        j = patchsize/2;

    elseif j + patchsize/2 > ny
        j = ny - patchsize/2;

    end

    SR1 = SR(gs1(i - patchsize/2 + 1 : i + patchsize/2, j - patchsize/2 + 1 : j + patchsize/2));
    SR2 = SR(gs2(i - patchsize/2 + 1 : i + patchsize/2, j - patchsize/2 + 1 : j + patchsize/2));

        if SR1 > SR2
        D_fin(cases(k)) = 1;

    elseif SR2 > SR1
        D_fin(cases(k)) = 0;

    else
        D_fin(cases(k)) = 0.5;

    end
end
toc 

figure; imshow(D_fin)

%step 4: fuse images
im_fin = Im1.*D_fin + (1-D_fin).*Im2;
figure; imshow(im_fin)

% Define functions used in script
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function grayscale = color2grayscale(I)

    %AS: taken from 'Detailed Fusion scheme' from assignment  (assumption is that channel numbers correspond to rgb)
    grayscale = 0.299*I(:,:,1) + 0.587*I(:,:,2) + 0.114*I(:,:,3);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function spatial_frequency = SR(I)
    
    %takes a grayscale 2D image and calculated the spatial frequency in the entire image.
    [N,M] = size(I);
    
    %definition of spatial frequency taken from assignment
    CF =  1/(N*M) * sum(sum( (I(2:end,:) - I(1:N-1,:)).^2 ));  %don't take the square root since it is squared in the next step anyways
    RF =  1/(N*M) * sum(sum( (I(:,2:end) - I(:,1:M-1)).^2 ));

    spatial_frequency = sqrt(CF + RF);
end