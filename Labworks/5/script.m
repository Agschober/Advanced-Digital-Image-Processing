% Progrmming assignment for AP3132-Advanced Digital Image Processing course
% Instructor: B. Rieger, F. Vos 
% Tutor: H. Heydarian
% Term: Q3-2021
%
% Labwork #5
%
clear all
close all
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Problem #1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Edge tracking
%%
% read the toy input image (comment/uncomment each block to generate the
% image)
sz = [200 200];

% toy image 1
% im = newim(sz);
% im(:,0:100) = 1;
% im = gaussf(im,8);

% toy image 2
% im = -xx(sz)+yy(sz)>0;
% im = gaussf(im,4);

% toy image 3
% im = -xx(sz)+yy(sz)>0;
% im(120:140,:) = 1;
% im = gaussf(im,8);

% toy image 4
im = -xx(sz)+yy(sz)>0;
im(120:170,100:end) = 1;
im = gaussf(im,8);

dipshow(im,'lin')
%%
% compute the image gradient
dX = dx(im);
dY = dy(im);

% TODO 1
% compute the cost C(r,c), do not use for loops here!
C = [];

% display the cost
dipshow(C,'lin')


C = im2mat(C); %convert for speed in the loops
A = zeros(sz);
for i=1:size(A,2)
    for j=2:size(A,1)
      %TODO 2-4
end
end

% select the minimum value on the last row of A
path = zeros(size(im,1),2); 
[minAval, minAid] = min(A(end, :));

% trace back the minimal cost path along A
for j=size(A,1):-1:1
    if minAid==1
        [minval_bt, minid_bt] = min([A(j,minAid), A(j,minAid+1)]);
        minAid = minAid + minid_bt - 1;
        path(j,1:2) = [j minAid];        
    elseif j==size(A,2)
        [minval_bt, minid_bt] = min([A(j,minAid-1), A(j,minAid)]);
        minAid = minAid + minid_bt - 2;
        path(j,1:2) = [j minAid];        
    else
        [minval_bt, minid_bt] = min([A(j,minAid-1), A(j,minAid), A(j,minAid+1)]);
        minAid = minAid + minid_bt - 2;
        path(j,1:2) = [j minAid];
    end
end

% plot the edge tracking result
dipshow(im,'lin');
hold on
line(path(:,2),path(:,1),'Color','red')

pause
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Problem #2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Mean shift image segmentation

% read the original input image
in = readim(['apple.jpg'])

% map NxMx3 image to MNx3 space (where M and N are height and width of the 
% image) to make a long vector of color information
in1 = im2mat(in,'double');
%in1 = shiftdim(in1,1); %this was needed before (older version of diplib) 
Ivec = reshape(in1, size(in,1)*size(in,2),3);

% scale the color to the range [0,1]
color = Ivec./255;

% plot the color information in 3D space

figure;
scatter3(Ivec(:,1), Ivec(:,2), Ivec(:,3), 50, color,'.') 
xlabel('color component');ylabel('color component');zlabel('color component')
axis equal
% check that this color plot looks like in the excerise

% apply meanshift to the vector of color infomation
radius = 30;    % radius of the serach window

[segmentMeans, pixelToSegments] = meanshift(Ivec', radius);

% remap the clusters to the color
myColorMap = segmentMeans';
clust = reshape(pixelToSegments, imsize(in,2),imsize(in,1));
figure; imshow(clust,[1,size(myColorMap,1)])
colormap(gca,myColorMap/255)
title('segmentation result')

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Problem #3
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 3-a Template matching

% read the original input image
in = readim('erika.tif');
h=dipshow(in);
h.Name='original image';
h.NumberTitle='off';

% read the template
T = readim(['eye.ics']);
dipshow(T);
h.Name='template';
h.NumberTitle='off';

% convert to Matlab array for speed in looping
T = double(T);
in = double(in);

% find both images sizes
[mI, nI] = size(in);
[mT, nT] = size(T);

% mean subtracted template
Tm = T - mean(T(:));

% corrolate both images
score = zeros(size(T));
for i=1:(mI-mT+1)
    for j=1:(nI-nT+1)
        
        % select the region of interest (sliding window)
        roi = in(i:i+mT-1, j:j+nT-1);
        
        % mean subtracted region of interest (ROI)
        % TODO 9
        roi = [];
        
        % compute the correlation
        % TODO 10
        corr = [];
        
        % normalize the correlation
        % TODO 11
        score(i,j) = [];
        
    end 
end

% plot the heatmap of the score
h=dipshow(mat2im(score),'lin');colormap('hot')
h.Name='heat map of the correlation';
h.NumberTitle='off';

% find the coordinates of the best match roi
[~, id] = max(score(:));
[matchIDX, matchIDY] = ind2sub(size(score), id);

% overlay the matched windows and the input image
h=dipshow(in);
hold on
rectangle('Position', [matchIDY, matchIDX, nT, mT], 'EdgeColor', [0 1 0]);
h.Name='matched patch';
h.NumberTitle='off';

%%
% 3-b Template matching (multiple instances, optional)
% Detect instances of the nuclear pore complex (NPCs)

% read the original input image
in = double((imread('npc_fov.tiff')));
dipshow(mat2im(in));
title('original image')

% read the template
T = double(rgb2gray(imread('npc_template.tiff')));
dipshow(mat2im(T));
title('template')

% find both images sizes
[mI, nI] = size(in);
[mT, nT] = size(T);

% mean subtracted template
Tm = T - mean(T(:));

% corrolate both images
score = zeros(size(T));

for i=1:(mI-mT+1)
    for j=1:(nI-nT+1)
        
        % select the region of interest (sliding window)
        roi = [];
        
        % TODO 12
        %{
        % just copy your code from previous section
        %}        
        
        % mean subtracted region of interest (ROI)
        roi =[]; 
        
        % compute the correlation
        corr = [];
        
        % normalize the correlation
        score(i,j) = [];
        
    end 
end

% plot the heatmap of the score
dipshow(mat2im(score),'lin');colormap('hot')
title('heat map of the correlation')


% TODO 13
%{
% extend the previsou code for detecting multiple instances at this point
% Hint: you do not have a single max score but multiples!
%}

peaks=0;

% plot the original image
dipshow(mat2im(in))
hold on

% overlay the matches
for i=1:peaks
    [matchIDX, matchIDY] = ind2sub(size(score), ids(i));    
    rectangle('Position', [matchIDY, matchIDX, nT, mT], 'EdgeColor', [0 1 0]);
end
title('matched patch')
