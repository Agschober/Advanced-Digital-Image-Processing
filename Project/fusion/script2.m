% code taken from [C. Liu et al. 'SIFT Flow: Dense Correspondence across Scenes and its Applications' IEEE trans. Pattern Analysis and Machine Intelligence]
% original code can be found here https://people.csail.mit.edu/celiu/ECCV2008/
% modified to recreate results from [Y. Liu et al. 'Multi-focus image fusion with dense SIFT' Information Fusion]
% for the course AP3132 Advanced Digital Image Processing at TU Delft in 2025 (assignment description can be found at https://qiweb.tudelft.nl/adip/projects/topic_07/)
% written by A. Schober & S. Verstraaten

% Step 0. Load and downsample the images
clear
close all

stack_path = '../focus_stack_adip/';
docs = dir(strcat(stack_path,'*.tif'));
nstack = size(docs,1);
filename = docs(1).("name");
im1 = imread(strcat(stack_path,filename));

[nx, ny, ncol] = size(im1);

clear im1;

scale = 50;
nx = round(nx/scale);
ny = round(ny/scale);

im_stack = zeros([nx,ny,ncol,nstack]);
gs_stack = zeros([nx,ny,nstack]);

tic
for i = 1:nstack
    im = imread(strcat(stack_path,docs(i).('name')));
    im_stack(:,:,:,i) = im2double(imresize(imfilter(im,fspecial('gaussian',7,1.),'same','replicate'),[nx, ny],'bicubic'));
    gs_stack(:,:,i) = color2grayscale(im_stack(:,:,:,i));
end
toc

clear im;

dipshow(gs_stack)

% Step 1. Compute the dense SIFT image

% patchsize is half of the window size for computing SIFT
% gridspacing is the sampling precision

patchsize=8;
gridspacing=1;

num_angles = 8;
num_bins = 4;

sift_stack = zeros([nx - (patchsize/2 +  2*gridspacing) ,ny - (patchsize/2 +  2*gridspacing),128,nstack]);
normalized_sift_stack = sift_stack;

for i = 1:nstack
    sift_stack(:,:,:,i) = dense_sift(gs_stack(:,:,i), patchsize, gridspacing);
    [nrows, ncols, cols] = size(sift_stack(:,:,:,1));
    sift_norm = reshape(sift_stack(:,:,:,i), [nrows*ncols num_angles*num_bins*num_bins]);
    sift_norm= normalize_sift(sift_norm);
    normalized_sift_stack(:,:,:,i) = reshape(sift_norm, [nrows ncols num_angles*num_bins*num_bins]);
end

clear sift_norm;

%{
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
%}

%Obtain activity maps
A = zeros([nx - 2*(patchsize/2 +  2*gridspacing),ny - 2*(patchsize/2 +  2*gridspacing),nstack]);

for i = 1:nstack
    sift = sift_stack(:,:,:,i);
    A(:,:,i) = sum(sift(patchsize/2:end-patchsize/2+1,patchsize/2:end-patchsize/2+1,:), 3);
end

clear sift;

gs_stack = gs_stack(patchsize/2:end-patchsize/2+1,patchsize/2:end-patchsize/2+1,:,:);
[nx, ny] = size(gs_stack(:,:,1));

%step 2: initial decision map
M = zeros(size(A));
D = M;

for k = 1:nstack
    for i = patchsize:nx - patchsize + 1
        for j = patchsize:ny - patchsize + 1
            s = sum(A(i - patchsize + 1: i, j - patchsize + 1: j,:), [1,2]);
            smax = squeeze(max(s));
            
            if size(smax) == 1
                index = find(s == smax);
                M(i - patchsize + 1: i, j - patchsize + 1: j, index) = M(i - patchsize + 1: i, j - patchsize + 1: j, index) + 1;
            end
        end
    end
end

%rough decision maps
for i = 1:nstack
    D(:,:,i) = sum(M, 3) - M(:,:,i) == 0;
end

dipshow(D)
pause

D_init = D1 + (1 - ((1-D2).*D1 + D2))*0.5;
figure;imshow(D_init)

%add post-processing step to D1 and D2
%remove small regions and holes in large areas (closing, opening or some other morphological operation)
threshold = round(nx*ny/100);
%D1 = closing(D1, round(nx/100));
%D2 = closing(D2, round(nx/100));

D_init = D1 + (1 - ((1-D2).*D1 + D2))*0.5;
%D_init = im2mat(D_init);
%figure;imshow(D_init)

%step 3: refine decision map using spatial frequency on image patch
cases = find(D_init == 0.5);

i_indices = 1:nx;
j_indices = 1:ny;
[i_indices, j_indices] = meshgrid(i_indices, j_indices);
i_indices = i_indices(cases);
j_indices = j_indices(cases);

D_fin = D_init;

tic
for k = 1:size(cases)
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
im_fin = Im1.*D_fin + (1-D_fin).*warpI2;
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



%
%% activity maps
%
%% memory
%
%% initial decision map
%
%% post processing
% 
%% refine decision map
%
%% fuse images
%
%
%
