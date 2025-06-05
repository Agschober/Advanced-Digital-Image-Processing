% code taken from [C. Liu et al. 'SIFT Flow: Dense Correspondence across Scenes and its Applications' IEEE trans. Pattern Analysis and Machine Intelligence]
% original code can be found here https://people.csail.mit.edu/celiu/ECCV2008/
% modified to recreate results from [Y. Liu et al. 'Multi-focus image fusion with dense SIFT' Information Fusion]
% for the course AP3132 Advanced Digital Image Processing at TU Delft in 2024/2025 (assignment description can be found at https://qiweb.tudelft.nl/adip/projects/topic_07/)
% written by A. Schober & S. Verstraaten

%% Importing image stack and pre processing (equal image sizes)
close all
stack_path = '../focus_stack_adip/';    % directory for image stack
docs = dir(strcat(stack_path,'*.tif')); % we used .tif files

%check whether there are such files in the directory
if isempty(docs)
    error('wrong file type chosen or no such files found in the specified path.');
end

nstack = size(docs,1);
filename = docs(1).("name");
im1 = imread(strcat(stack_path,filename));  
[nx_orig, ny_orig, ncol_orig] = size(im1);  
clear im1;

%Scale down images (for memory and speed)
scale = 5;
nx = round(nx_orig/scale);
ny = round(ny_orig/scale);

im_stack = struct([]);
color_stack_for_display = zeros(nx, ny, 3, nstack, 'like', im2double(imread(strcat(stack_path, docs(1).('name')))));
gs_stack =  zeros(nx, ny, nstack, 'like', im2double(imread(strcat(stack_path, docs(1).('name')))));

disp('reading image stack')
tic
for i = 1:nstack
    current_filename = docs(i).('name');
    im = imread(strcat(stack_path, current_filename));
    im = imfilter(im, fspecial('gaussian', 7, 1.0), 'same', 'replicate');
    im = imresize(im, [nx, ny], 'bicubic');
    
    color_stack_for_display(:,:,:,i) = im2double(im);
    gs_stack(:,:,i) = color2grayscale(color_stack_for_display(:,:,:,i)); 
end
toc
clear im current_filename;

%display stack
dipshow(joinchannels('RGB',color_stack_for_display(:,:,1,:), color_stack_for_display(:,:,2,:), color_stack_for_display(:,:,3,:)));



%% Step 1. Compute the dense SIFT image
% patchsize is half of the window size for computing SIFT
% gridspacing is the sampling precision
patchsize= 8;
gridspacing= 1;
num_angles = 8;
num_bins = 4;

disp('creating sift images')
tic
for i = 1:nstack
    [im_stack(i).sift, grid_x, grid_y] = dense_sift(gs_stack(:,:,i), patchsize, gridspacing);
    [nrows, ncols, cols] = size(im_stack(i).sift);
    sift_norm = reshape(im_stack(i).sift, [nrows*ncols num_angles*num_bins*num_bins]);
    sift_norm = normalize_sift(sift_norm);
    im_stack(i).normalized_sift = reshape(sift_norm, [nrows ncols num_angles*num_bins*num_bins]);
end
toc
clear sift_norm;



%% step 2. Obtain activity maps
disp('obtaining activity scores')
tic
A = struct;

for i = 1:nstack
    sift = im_stack(i).sift;
    A(i).amap = sum(sift, 3);
end
clear sift;
toc

gs_stack = gs_stack(patchsize/2:end-patchsize/2+1,patchsize/2:end-patchsize/2+1,:);
[nx, ny] = size(gs_stack(:,:,1));



%% Step 3. Initial decision map
disp('obtaining initial decision map')
tic
decision_radius = 2*patchsize;

M = zeros(nx,ny,nstack);

s = zeros(nstack, 1);
for i = decision_radius:nx
    for j = decision_radius:ny
        for k = 1:nstack
            patch = A(k).amap(i - decision_radius + 1: i, j - decision_radius + 1: j);
            s(k) = sum(patch, 'all');
        end

        [~, index] = max(s);
        M(i - decision_radius + 1: i, j - decision_radius + 1: j,index) = ...
            M(i - decision_radius + 1: i, j - decision_radius + 1: j,index) + 1;
    end
end

D = false(size(M));

for i = 1:nstack
    D(:,:,i) = (sum(M, 3) - M(:,:,i)) == 0;
end
toc

dipshow(D);



%% Step 4. post-processing on decision map
n_filt = 1;
dim_filt_x = ceil(nx / 10); %filter size is 100th the size of the source image
dim_filt_y = ceil(ny / 10);
filt_type = 'rectangular';
disp('applying post processing step')
tic
for i = 1:n_filt
    for j = 1:nstack
        D_tmp = D(:,:,j);
        D(:,:,j) = medif(D_tmp, [dim_filt_x, dim_filt_y], filt_type);
    end
end
toc
dipshow(D);
clear D_tmp



%% Step 6. Uncertainty mask
disp('decision map refinement')
tic
mask = sum(D, 3) == 0;
dipshow(mask)
cases = find(mask);

i_indices = 1:nx;
j_indices = 1:ny;
[j_indices, i_indices] = meshgrid(j_indices,i_indices);
i_indices = i_indices(cases);
j_indices = j_indices(cases);

for k = 1:size(cases,1)
    i = i_indices(k);
    j = j_indices(k);

    if i - patchsize/2 < 0
        i = patchsize/2;
    elseif i + patchsize/2 > nx
        i = nx - patchsize/2 -1;
    end

    if j - patchsize/2 < 0
        j = patchsize/2;
    elseif j + patchsize/2 > ny
        j = ny - patchsize/2 - 1;
    end

    SF_stack = SF(gs_stack(i - patchsize/2 + 1 : i + patchsize/2, j - patchsize/2 + 1 : j + patchsize/2,:));
    [SF_max, index] = max(SF_stack);
    D(i,j,index) = 1 / numel(index);
  
end
toc

clear gs;

dipshow(D)



%% Step 7: Fuse images
color_stack_for_display = color_stack_for_display(patchsize/2:end-patchsize/2+1,patchsize/2:end-patchsize/2+1,:, :);

im_fin = sum(gs_stack .* D, 3);
color_im_fin = zeros(size(im_fin, 1),size(im_fin, 2),3);
for i = 1:3
    color_im_fin(:,:,i) = sum(squeeze(color_stack_cropped(:,:,i,:)) .* D, 3);
end

% Display the fused image
dipshow(im_fin)
dipshow(joinchannels('RGB', color_im_fin(:,:,1), color_im_fin(:,:,2), color_im_fin(:,:,3)))


% Define functions used in script
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function grayscale = color2grayscale(I)
    % AS: taken from 'Detailed Fusion scheme' from assignment
    grayscale = 0.299*I(:,:,1) + 0.587*I(:,:,2) + 0.114*I(:,:,3);
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function spatial_frequency = SF(I)
    % Takes a grayscale 2D image and calculates the spatial frequency
    [N,M,~] = size(I);
    CF =  1/(N*M) * sum((I(2:end,:,:) - I(1:N-1,:,:)).^2, [1,2]);
    RF =  1/(N*M) * sum((I(:,2:end,:) - I(:,1:M-1,:)).^2, [1,2]);
    spatial_frequency = sqrt(CF + RF);
end

