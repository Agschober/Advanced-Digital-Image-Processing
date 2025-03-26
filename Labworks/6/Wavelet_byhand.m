% Progrmming assignment for AP3132-Advanced Digital Image Processing course
% Instructor: B. Rieger, F. Vos 
% Tutor: H. Heydarian
% Term: Q3-2020
%
% Labwork Wavelet% compute the wavelet transform graphically (just by pen and paper)
close all

f = [1 4 -3 0];
xpos = [0.5 1.5 2.5 3.5];
figure;
bar(xpos,f,1)

% the scaling function of the Haar wavelet:
% you just take the average between neighboring function values and repeat
% the process until only the average value of the signal remains
s1 = [2.5 2.5 -1.5 -1.5]; %2 values
s2 = [0.5 0.5 0.5 0.5];   %1 value

hold on
bar(xpos,s1,'r')
bar(xpos,s2,'m')
legend('signal','scale 1','scale 2')
hold off

% for the approximation functions you take the difference between the
% different level of scaling functions.

a1 = f-s1;
a2 = s1-s2;

figure;bar(xpos,a1,1);title('a1')
figure;bar(xpos,a2,1);title('a2')

% from this you can write down the wavelet transformation if you take into
% account the scaling factor 2^(-i/2) for the de Haar wavelet (the factor
% 2^-i/2 ensures orthonormality of the wavelets

%looking at s2 you see a value of 0.5 which gives a wavelet coefficient of
% 0.5
% looking at a2 you see value of 2 for the wavelet
% looking at a1 you see values of -1.5 and -1.5 for the wavelet
% 

haar_wavelet(f)
