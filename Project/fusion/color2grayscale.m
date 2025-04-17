function grayscale = color2grayscale(I)

    %AS: taken from 'Detailed Fusion scheme' from assignment  (assumption is that channel nummers correspond to rgb, i believe this is fair)
    grayscale = 0.299*I(:,:,1) + 0.587*I(:,:,2) + 0.114*I(:,:,3);