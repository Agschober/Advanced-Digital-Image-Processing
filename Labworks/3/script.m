% Progrmming assignment for AP3132-Advanced Digital Image Processing course
% Instructor: B. Rieger, F. Vos 
% Tutor: H. Heydarian
% Term: Q3-2021
%
% Labwork #3
%
clear, close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Problem #1
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Part a: dilation and erosion

% read the original input image
I = readim('gold.tif');
h=dipshow(I);
h.Name='original image';
h.NumberTitle='off';

	% add proper noise to make the image the same as the one depicted in the 
% manual (top-right in Figure 1)
% TODO3 AS: this was already done apparantly
J = noise(I,'saltpepper',0,0.1);
h = dipshow(J);
h.Name='noisy image';
h.NumberTitle='off';

% apply dilation to the noisy image
In = J;
I_max = max_f(In, 3);
h=dipshow(I_max);
h.Name='dilated image';
h.NumberTitle='off';

% apply erossion to the noisy image
I_min = min_f(In, 3);
h=dipshow((I_min));
h.Name='eroded image';
h.NumberTitle='off';

pause
%%
% Part b: closing and opening

% apply closing to the given image
% TODO4a
bsize = 5; %AS: for consistency in structuring element size for opening and closing (repeated operations)

function out = max_min_f(in, bsize)
    out = min_f(max_f(in, bsize), bsize);
end

function out = min_max_f(in, bsize)
    out = max_f(min_f(in, bsize), bsize);
end

I_close = max_min_f(I, bsize); %AS: closing is erosion then dilation, opening vice versa
h=dipshow(I_close);
h.Name='closing';
h.NumberTitle='off';

% apply openning to the given image
% TODO4b AS: just change order of erosion and dilation
I_open = min_max_f(I, bsize);
h=dipshow(I_open);
h.Name='opening';
h.NumberTitle='off';


%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Problem #2
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Part a: smoothing

% read the original input image
I = readim(['trui.tif']);
h=dipshow(I);
h.Name='original image';
h.NumberTitle='off';

% TODO5
% add proper noise to make the image as depicted in the lecture notes
J = noise(I,'gaussian',10); %AS: you can see the smoothed image gets affected once the standard deviation is about 10
h=dipshow(J);
h.Name='noisy image';
h.NumberTitle='off';

In = J;
bsize = 3; %AS: for easy adjustment of SE size
I_max = max_f(In, bsize); 
I_min = min_f(In, bsize);

function out = dyt(in, bsize)
    out = 1/2*(max_f(in, bsize) + max_f(in, bsize));
end

function out = tet(in, bsize)
    out = 1/2*(max_min_f(in, bsize) + min_max_f(in, bsize));
end

Iout1= dyt(I, bsize); %AS: method 1, average of erosion and dilation -> dynamic threshold
h=dipshow(Iout1);
h.Name='smoothed image (method1)';
h.NumberTitle='off';

Iout2= tet(I, bsize); %AS: method 2, average of opening and closing -> texture threshold
h=dipshow(Iout2);
h.Name='smoothed image (method2)';
h.NumberTitle='off';

pause

Iout3= dyt(dyt(I, bsize), bsize); %AS: method 1, average of erosion and dilation -> dynamic threshold
h=dipshow(Iout3);
h.Name='twice smoothed image (method1)';
h.NumberTitle='off';

Iout3=tet(tet(I, bsize), bsize); %AS: method 2, average of opening and closing -> texture threshold
h=dipshow(Iout3);
h.Name='twice smoothed image (method2)';
h.NumberTitle='off';

pause
%%
% Part b: image gradient

% read the original input image
I = readim(['truinoise.tif']);
h=dipshow(I);
h.Name='original image';
h.NumberTitle='off';

% TODO6
% structuring element size
k = 5;

% TODO6a
% dilation
% erosion
% openning
% closing
function out = dyr_max_min(in, bsize)
    out = max_f(in, bsize) - min_f(in, bsize);
end

function out = ter_max_min(in, bsize)
    out = max_min_f(in,bsize) - min_max_f(in, bsize);
end

% image gradient method 1 TODO6b
dyr = dyr_max_min(I,k); 
h=dipshow(dyr);
h.Name='gradient  dyr';
h.NumberTitle='off';

% image gradient method 2 TODO6c
ter = ter_max_min(I, k);
h=dipshow(ter);
h.Name='gradient  ter';
h.NumberTitle='off';

% image gradient method 3 TODO6d
rar = [dyr - ter];
h=dipshow(rar);
h.Name='gradient image rar';
h.NumberTitle='off';

% compare with gradmag() from diplib
grad = gradmag(I);
h = dipshow(grad);
h.Name = 'gradmag() from diplib'
h.NumberTitle= 'off';
% Part c: Background removal

% read the original input image
I = readim(['retinaangio.tif']);
h=dipshow(I);
h.Name='original image';
h.NumberTitle='off';

% TODO7
bg=[max_min_f(I, 27)];
nobg= [I - bg];

h=dipshow((bg));
h.Name='background';
h.NumberTitle='off';

h=dipshow((nobg));
h.Name='background removed';
h.NumberTitle='off';
