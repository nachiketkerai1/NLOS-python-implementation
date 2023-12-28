close all
clear
clc

%{
    Description:
    Unoptimized Implementation for paper: 
    Phasor Field Diffraction Based Reconstruction for Fast Non-Line-of-Sight Imaging Systems
    
    The code starts from reading Fourier domain histogram (FDH, wavefront cube) in the "input"
    folder, then perform fast Rayleigh Sommerfeld Diffraction (RSD)
    algorithm for the reconstruction.

    Notice that one can generate Fourier domain histogram from time binning
    histogram, we provide a script in the folder named
    "generate_fourierDhist.m" which use calibrated time binning histogram
    to generate Fourier domain histogram for reconstruction.

    This code performs reconstructions with  "spatial sectioning" method
    which can be used to evalulate large field of view in the hidden scene

%}

%% Dataset id
id = 6; % 6-10

%% parameters
tag_maxdepth = 2; % signal of interest respect to the targets, unit meter, maximum target depth (furthest target)
depth_min = 0.5; % depth minimal volumn
depth_max = 1.5; % depth maximum volumn
v_apt_Sz = 2; % virtual aperture maximum size, automatical pad to symmetry (square), unit meter, standard size 2
d_level = 2; % number of disjoint piece in the reconstruction space
d_wcycle = 7; % cycle for the illumination pulse, should be adjusted if changinag the illumination temporal window
d_offset = 0.1; % electrical delay as an constant offset 

%% loading dataset
switch id
    case {6}
        str_name = 'officescene';   
        tag_maxdepth = 3;
        v_apt_Sz = 3;
        depth_max = 2.5;
    case {7}
        str_name = 'officescene_corrected_1ms'; 
        tag_maxdepth = 3;
        v_apt_Sz = 3;
        depth_max = 2.5;
        d_wcycle = 5; 
        d_offset = 0; 
    case {8}
        str_name = 'officescene_corrected_5ms';  
        tag_maxdepth = 3;
        v_apt_Sz = 3;
        depth_max = 2.5;
        d_wcycle = 5; 
        d_offset = 0; 
    case {9}
        str_name = 'officescene_corrected_10ms'; 
        tag_maxdepth = 3;
        v_apt_Sz = 3;
        depth_max = 2.5;
        d_wcycle = 5; 
        d_offset = 0; 
    case {10}
        str_name = 'officescene_corrected_20ms'; 
        tag_maxdepth = 3;
        v_apt_Sz = 3;
        depth_max = 2.5;
        d_wcycle = 5; 
        d_offset = 0; 
end 

load(['./input/' str_name '_' num2str(tag_maxdepth) '.mat']);

%% Basic parameters for wavefront cube
c_light = 299792458; % define speed of light
sample_spacing = aperturefullsize(1)/(size(u_total,1) - 1); % re-calculate the sampling space

% Pad Virtual Aperture Wavefront
% additional if physical dimension is odd number
if(mod(size(u_total,1),2)==1)
    [tmp_x, tmp_y, tmp_z] = size(u_total);
    tmp3D = zeros(round(tmp_x/2)*2, round(tmp_y/2)*2, tmp_z);
    aperturefullsize = [size(tmp3D,1)-1, size(tmp3D,2)-1] * sample_spacing;
    tmp3D(1:size(u_total,1), 1:size(u_total,2),:) = u_total;
    u_total = tmp3D;
end

[~, ~, SPAD_index]= Create_VirtualAperture(squeeze(u_total(:,:,1)), aperturefullsize, v_apt_Sz, SPAD_index); % update SPAD index 

% perallocated memeory for padding wavefront, fix bug for any sampling
% spacing
u_tmp = zeros(2 * round(round(v_apt_Sz/(sample_spacing))/2), 2 * round(round(v_apt_Sz/(sample_spacing))/2), size(u_total,3));

for index = 1 : size(u_total,3)
    tmp = squeeze(u_total(:,:,index));
    [u_tmp(:,:,index), apt_tmp]= Create_VirtualAperture(tmp, aperturefullsize, v_apt_Sz, SPAD_index);
end

aperturefullsize = apt_tmp; % update virtual aperture size
u_total = u_tmp; % update wavefront cube

% Spatial downsampling by bounded resolution 
u_total = u_total(1:2:end, 1:2:end, :) + u_total(2:2:end, 2:2:end, :);
SPAD_index = SPAD_index/2; % update SPAD index

% create depth based on virtual sensor re-sampling resolution
depth_loop = depth_min : 2 * sample_spacing: depth_max;

%% Frequency backpropagation
nZ = length(depth_loop);

u_volume = zeros([size(u_total,1), size(u_total,2), nZ]);
tmp_volumn = zeros([size(u_total,1), size(u_total,2), nZ]);

% calculate the disjoint piecewise mask for large FOV setting
central_lambda = lambda_loop(ceil(end/2),:);
[d_offset, d_mask] = Create_DisDiffCube(SPAD_index, u_volume, depth_min, 2 * sample_spacing, d_wcycle*central_lambda, d_offset, d_level);

%% Reconstruction
display('Reconstruction ... ...');
tic

for index = 1 : size(d_mask,4)
    tmp_offset = d_offset(index);
    tmp_mask = fliplr(squeeze(d_mask(:,:,:,index)));
    
    for i = 1 : length(depth_loop)
        depth = depth_loop(i);
        u_tmp = zeros(size(u_total,1), size(u_total,2)); 

        for spectrum_index = 1 : length(lambda_loop)
            u_field = squeeze(u_total(:,:,spectrum_index));
            lambda = lambda_loop(spectrum_index);
            omega = omega_space(spectrum_index);

            % Define time of arrival and apply fast RSD reconstruction formula
            u1 = u_field.*exp(1j * omega * (depth + tmp_offset)/c_light * ones(size(u_field)));
            u2_RSD_conv = Camera_Focusing(u1, aperturefullsize, lambda, depth, 'RSD convolution', 0);

            u_tmp = u_tmp + weight(spectrum_index) * u2_RSD_conv;

        end
        tmp_volumn(:,:,i) = u_tmp;
    end
    
    % merge FOV sub images
    u_volume = u_volume + tmp_volumn.*tmp_mask; 
end

toc_clock = toc;
display(sprintf(['Reconstruction in %f seconds, ... done'], toc_clock));

%% Visualization part
mgn_volume = abs(u_volume);

% Maximum projection along depth direction
[img, ~] = max(mgn_volume,[],3);

img = imresize(img,2);


figure;
imagesc(img);
colormap 'hot';
axis image;
axis off;

%% Alternative functions

%{
    Parameters input: 
        1. uin: input virtual wavefront
        2. L: aperture size, unit meter
        3. lambda: virtual wavelength, unit meter
        4. depth: propagation (focusing) depth, unit meter
        5. method: string for choosing variety wave propagation kernel 
        6. alpha: coefficient for accuracness, ideally >= 1, 0.1 is
        acceptable for small amplitude error (large phase error)

    Parameters output: 
        1. uout: output wavefront at the destination plane
%}
function[uout] = Camera_Focusing(uin, L, lambda, depth, method, alpha)
    if (alpha~= 0)
        [u1_prime, aperturefullsize_prime, pad_size, Nd] = tool_fieldzeropadding(uin, L, lambda, depth, alpha);
    else
        u1_prime = uin;
        aperturefullsize_prime = L;
        pad_size = 0;
        Nd = size(u1_prime,1);
    end
    
    switch method      
        case 'RSD convolution'
            uout = fliplr(propRSD_conv(u1_prime, aperturefullsize_prime, lambda, depth));
        
    end
    uout = uout(pad_size+1: pad_size+Nd, pad_size+1: pad_size+Nd);

% Padding the Wavefront
function[u2, phyoutapt, pad_size, sM] = tool_fieldzeropadding(u1, phyinapt, lambda, depth, alpha)
    % % Input: 
    % u1: input complex field
    % phyinapt: Input physical aperture dimension
    % lambda : wavelength, unit m
    % depth: unit m
    % alpha: uncertainty parameter
    % % Output: 
    % u2: output complex field
    % phyoutapt: Output physical aperture dimension
    % pad_size: one sided padding size
    % sM: ola square symmetry, parameter for redoing the padding

    % This program input x-y sampling interval is the same, this program
    % also check if input field is symmetry (square)
    
    [M, N] = size(u1);
    delta = phyinapt(1)/(M-1);
    
    if(M == N)

    else
        [u1, ~] = tool_symmetry(u1, phyinapt);
    end
    
    [sM, ~] = size(u1); % update new discret index, sM output for redo the padding
    
    N_uncertaint = (lambda * abs(depth))/(delta^2);
    pad_size = (N_uncertaint - sM)/2;

    if(pad_size > 0)
        pad_size = round(alpha * pad_size);
    else
        pad_size = 0;
    end
    
    u2 = padarray(u1, [pad_size,pad_size], 0);
    phyoutapt = delta * (size(u2)-1);
    
    function[u2, phyoutapt] = tool_symmetry(u1, phyinapt)
        % % Input: 
        % u1: input complex field
        % phyinapt: Input physical aperture dimension
        % % Output: 
        % u2: output complex field
        % phyoutapt: Output physical aperture dimension

        % This program input x-y sampling interval is the same

        [M, N] = size(u1);
        delta = phyinapt(1)/(M-1);

        if(M > N)    
            square_pad = ceil(0.5 * (M - N));
            u2 = padarray(u1, [0,square_pad],0);
            phyoutapt = delta.*size(u2);
        elseif (M < N)
            square_pad = ceil(0.5 * (N - M));
            u2 = padarray(u1, [square_pad,0],0);
            phyoutapt = delta.*size(u2);
        else
            u2 = u1;
            phyoutapt = phyinapt;
        end

    end
end


% RSD convolution Focusing
function[u2] = propRSD_conv(u1, L,lambda, z)
	% RSD in the convolution foramt
    
	% assumes same x and y side lengths and uniform sampling
	% u1 - source plane field
	% L - source and observation plane side lengths
	% lambda - wavelength
	% z - propagation distance
	% u2 - observation plane field
    
    [M,N] = size(u1);			% get input field array size, physical size

    % Extract each physical dimension
    l_1 = L(1);
    l_2 = L(2);
     
    % Spatial sampling interval
    dx = l_1/(M-1);						% sample interval x direction
    dy = l_2/(N-1);						% sample interval y direction

    % spatial sampling resolution needs to be equal in both dimension
    z_hat = z/dx;                   
    mul_square = lambda * z_hat/(M * dx);

    % center the grid coordinate
    m = linspace(0, M-1, M); m = m - M/2;
    n = linspace(0, N-1, N); n = n - N/2;
    
    [g_m, g_n] = meshgrid(n,m);  % coordinate mesh			

    %  Convolution Kernel Equation including the firld drop off term
    h = exp(1j * 2 * pi * (z_hat.^2 * sqrt(1 + (g_m.^2/z_hat.^2 + g_n.^2./z_hat.^2)) ./(mul_square * M)))./sqrt(1 + g_m.^2/z_hat^2 + g_n.^2/z_hat^2);  
        
    % Convolution or multiplication in Fourier domain
    H = fft2(h);
    U1 = fft2(u1);
    U2 = U1.*H;
    u2 = ifftshift(ifft2(U2));
end

end

%{
    Parameters input: 
        1. uin: captured virtual wavefront, unitless
        2. apt_in: captured physical aperture sizem, uniter meter (array x-y)
        3. maxSz: virtual aperture physical maximum size, unit meter (real valed scalar)
        4. posL: virtual point source location on the aperture (origional
        physical captured) index [x y] unitless

    Parameters output: 
        1. uout: output virtual wavefront, square size, unitless
        2. apt_out: output virtual aperture physical maximum size, unit meter (array x-y)
        3. posO: output new virtual point source location on the apeture,
        index [x y] unitless

    Description:
        This function works for creating virtual aperture, independent from
        RSD propagation. Input can be symmetry or not, output dimension
        follow by the input sampling resolution and the maxSz requirement.
        
        Padding process refers to the center of the captured wavefront
        center.
%}
function [uout, apt_out, posO] = Create_VirtualAperture(uin, apt_in, maxSz, posL)
    % Get input wavefront dimension
    [M, N] = size(uin);
    
    % Calculate the spatial sampling density, unit meter
    delta = apt_in(1)/(M-1);
    
    % Calculate the ouptut size
    M_prime = round(maxSz/delta);
    
    % symmetry square padding
    if(M == N)

    else
        [uin, ~] = tool_symmetry(uin, apt_in);
    end
    
    % Padding size difference
    diff = size(uin) - [M, N] ;
    posL = posL + diff/2; % update the virtual point source location after symmetry
    
    % Update the symmetry aperture size
    [M, ~] = size(uin);
    
    % Virtual aperture extented padding on the boundary
    % difference round to nearest even number easy symmetry padding
    dM = 2 * ceil((M_prime - M)/2);
    
    % symmetry padding on the boundary
    if(dM > 0)
        % Using 0 boundary condition
        uout = padarray(uin, [dM/2 dM/2], 0, 'both');
        posO = posL + dM/2; % update the virtual point source location after padding
    else
        uout = uin;
        posO = posL;
    end
    
    % update the virtual aperture size
    apt_out = delta * size(uout);
    
    
function[u2, phyoutapt] = tool_symmetry(u1, phyinapt)
    % % Input: 
    % u1: input complex field
    % phyinapt: Input physical aperture dimension
    % % Output: 
    % u2: output complex field
    % phyoutapt: Output physical aperture dimension

    % This program input x-y sampling interval is the same

    [M, N] = size(u1);
    delta = phyinapt(1)/(M-1);

    if(M > N)    
        square_pad = ceil(0.5 * (M - N));
        u2 = padarray(u1, [0,square_pad],0);
        phyoutapt = delta.*size(u2);
    elseif (M < N)
        square_pad = ceil(0.5 * (N - M));
        u2 = padarray(u1, [square_pad,0],0);
        phyoutapt = delta.*size(u2);
    else
        u2 = u1;
        phyoutapt = phyinapt;
    end

end
    
end

%{
    Parameters input: 
        1. indexL: virtual illumination source index on the refers to the aperture matrix, [x y]
        2. sDim: dimension of the output reconsruction volumn cube
        3. dmin: 1st layer in Z direction physical depth minimal value, unit meter
        4. res: voxel resolution, currently equals to the spatial sampling spacing
        5. pulseLen: synethsis pulse length, unit meter
        6. initial_offset: initital offset, default is 0, should based on
        single source locations and volumn to adjust
        7. dlevel: distance level (time shift approx resolution), unitless (integer)
        
    Parameters output: 
        1. DoffsetVal: distance offset level, will be used in the RSD backpropagation
        2. DMask: corresponding spatial binary mask, will be used in the wavefront merging process 

    Description:
        This function is to apply piecewise approximated plane wavefront
        for the reconstruction of the complex scene

%}
function [DoffsetVal, DMask] = Create_DisDiffCube(indexL, sDim, dmin ,res, pulseLen, initial_offset, dlevel)
% function [discube, disdiffcube] = Create_DisDiffCube(indexL, sDim, dmin ,res, pulseLen, dlevel)
    
    % Get output Volumn dimension
    [nX, nY, nZ] = size(sDim);
    
    % Percomputed timeshift volumn
    discube = zeros(nX, nY, nZ);
    disdiffcube = zeros(nX, nY, nZ);
    
    % Virtual Light source location on the aperture in Matrix format
    SPAD_index = indexL; % indexing location on the aperture plane
    
    % Calculate the distance shift from the point source
    for i = 1 : nX
        for j = 1 : nY
            for k = 1 : nZ
                pt_plane = [i j];
                d_r = res * norm(pt_plane - SPAD_index);
                d_v = dmin + res * (k-1);
                d_tmp = sqrt(d_r.^2 + d_v.^2);
                % Ideal distance shift from a single point source
                discube(i,j,k) = d_tmp;
                 % Calculate the plane wavefront approximation error
                disdiffcube(i,j,k) = d_tmp - d_v;
            end
        end
    end
    
    % Create distance level offset
    DoffsetVal = initial_offset + pulseLen * (0 : dlevel-1); % calculate the offset level, assign the center on the interval as the offset

 
    % Create logical mask corresponding to the distance offset value array
    DMask = zeros(nX, nY, nZ, dlevel); % per-allocated mask volumn, x-y-z-offsetlevel
    
    for i = 1 : size(DMask,4)
        DMask(:,:,:, i) = (disdiffcube > ((i-1) * pulseLen)).* (disdiffcube <= (i * pulseLen));
    end
    
end


