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

    This code performs generating Fourier domain histogram from time
    binning data. The binning time response is in folder './tdata/'
    Then save the FDH in the folder './input/'. Notice that, FDH can be captured directly.
    
    time binning data in the "./tdata/" folder can be download from opensource link:
    https://biostat.wisc.edu/~compoptics/
    when used data please cite following paper as well: 
    Liu, Xiaochun, et al. "Non-line-of-sight imaging using phasor-field virtual wave optics." Nature 572.7771 (2019): 620-623.
    
%}

%% Dataset id
id = 1; % 1-10

%% Common Defaule parameters
z_gate = 500; % digital gate index, remove gating artifacts
K = 2; % temporal downsampling coefficient 
sc = 2; % virtual wave sampling coefficient
tag_maxdepth = 2; % maximum target depth (furthest target)
ill_cycle = 5; % temporal virtual window length based on cycle 
peak_ratio = 1e-1; % illumination function in Fourier domain coefficient threshold ratio based on peak value

%% loading dataset
switch id
    case {1}
        str_name = 'letter4'; % 0.9
        tag_maxdepth = 1.5;
    case {2}
        str_name = 'resolutionbar'; % 0.73
        tag_maxdepth = 1.5;
    case {3}
        str_name = 'NLOSletter'; % 0.75
        tag_maxdepth = 1.5;
    case {4}
        str_name = 'shelf_targets_lighton'; % 0.83
        tag_maxdepth = 1.5;
    case {5}
        str_name = 'letter44i'; % multidepth  
        tag_maxdepth = 1.5;
    case {6}
        str_name = 'officescene'; % multidepth  
        tag_maxdepth = 3;
    case {7}
        str_name = 'officescene_corrected_1ms'; % multidepth 
        tag_maxdepth = 3;
        sc = 3; 
        z_gate = 700;
        ill_cycle = 4; % more cycle to cancel the noise
    case {8}
        str_name = 'officescene_corrected_5ms'; % multidepth  
        tag_maxdepth = 3;
        sc = 3;
        z_gate = 700;
    case {9}
        str_name = 'officescene_corrected_10ms'; % multidepth  
        tag_maxdepth = 3;
        sc = 3;
        z_gate = 700;
    case {10}
        str_name = 'officescene_corrected_20ms'; % multidepth  
        tag_maxdepth = 3;
        sc = 3;
        z_gate = 700;
end


load( ['./tdata/performat_' str_name '.mat' ]); 

rect_data(:,:, 1 : z_gate) = 0; % gate artifact, 500 works for all the dataset

[M,N,T] = size(rect_data);  % sptial X, sptial Y, temporal T
fs = 1/ts; % sampling frequency, we did not use this
c_light = 299792458; % speed of light
binresolution = ts * c_light; % unit meter

%% Temporal Down sampling 
for k = 1:K
    T = T./2;
    binresolution = 2*binresolution;
    ts = 2*ts;
    fs = fs/2;
    rect_data = rect_data(:,:,1:2:end) + rect_data(:,:,2:2:end);
end


%% Virtual Illumination Parameters
lambda = 2 * sampling_spacing * sc; % unit meter, virtual wavelength

% Synethsis pulse
t_ill = Illumination_kernel(lambda, ill_cycle*lambda, ts, 'gaussian'); % * cycle respect to the virtual wavelength
t_ill = Illumination_pad(t_ill, tag_maxdepth, ts); 

% Trucation data based on the illumination function
rect_data = rect_data(:,:,1:length(t_ill));

% Illumination Fourier Series Decomposition
[fre_mask, lambda_loop, omega_space, weight] = Illumination_decomp(t_ill, ts, peak_ratio);
num_component = length(omega_space);

% Total wavefrotn cube
u_total = zeros(M, N, num_component);

%% Generated FDH from captured time response
display('Loading data ... done');

tic
for ii = 1 : size(rect_data,1)
    for jj = 1 : size(rect_data,2)
        t_tmp = squeeze(rect_data(ii, jj, :));
        f_tmp = fft(t_tmp);
        f_slice = f_tmp(fre_mask);
        u_total(ii,jj,:) = f_slice;
    end
end
toc_clock = toc;

display(sprintf(['Generate FDH from captured time response in %f seconds'], toc_clock));

%% Saving the FDH wavefront cube
display('Saving FDH ... ...');
save(['./input/' str_name '_' num2str(tag_maxdepth) '.mat'], 'u_total', 'weight', 'lambda_loop', 'omega_space', 'aperturefullsize', 'sampling_spacing', 'SPAD_index', '-v7.3');
display('done');


%% Alternative function used for this script
function[frequency_index, L, O, W] = Illumination_decomp(t, ts, ratio)
    %{
    Illumination Frequency Decomposition
    Parameters input: 
        1. t: temporal illumination kernel function
        2. ts: temporal sampling rate, unit second
        3. ratio: ratio of the energy coefficient for the fitering step

    Parameters output: 
        1. frequency_index: frequency mask
        2. l: lambda sequence 
        3. o: omega sequence
        4. w: linear weight coefficient
%}
    c_light = 299792458; % speed of light
    
    % Calculate frequency sampling
    fs = 1/ts;
    
    % Get length of the signal
    N = length(t); % N refers to the total discrete sample 
   
    % Energy for the illumination kernel in Fourier domain
    P_wave = abs(fft(t)/N);
    P_wave = P_wave(1:round(N/2+1)); 
    
    % Calculate the ratio based on the Energy peak value
    coeff_ratio = P_wave./max(P_wave(:));
    
    % Find  the filtered frequency index (only index, no unit associate)
    frequency_index = find(coeff_ratio>=ratio); 
    
    % Calculate the final synethsis quantity
    F = fs.*frequency_index./N;
    O = 2 * pi * F;
    L = c_light * 2 * pi ./ O;
    W = P_wave(frequency_index); % weight from the gaussian spectrum
    
end

function[kernel] = Illumination_kernel(lambda, length, ts, type)
%{
    Parameters input: 
        1. lambda: virtual wavelength, unit meter
        2. length: length of the signal, unit meter
        3. ts: temporal samplingm unit second
        4. type: type of the illumination kernel

    Parameters output: 
        1. kernel: output 1d illumination kernel
%}
    c_light = 299792458; % speed of light
    bin_res = c_light * ts; % map temporal sampling to spatial meter, unit m
    
    switch type
        case 'gaussian'
            v_s = (lambda/bin_res);
            length = round(length/bin_res);
            sin_pattern = length/v_s; 
            
            cmp_sin = exp(1j * 2 * pi * (sin_pattern * linspace(1,length,length)')/length);
            gauss_wave = gausswin(length, 1/0.3);
            
            kernel = cmp_sin.*gauss_wave;
    end
end

function[t_out] = Illumination_pad(t_in, Maxdepth, ts)
%{
    Parameters input: 
        1. t_in: temproal signal input
        2. Maxdepth: unit meter, maximum depth for signal of interest in reconstruction
        3. ts: temporal sampling rate, unit second

    Parameters output: 
        1. kernel: output 1d illumination kernel
%}
    c_light = 299792458; % speed of light
    bin_res = c_light * ts; % map temporal sampling to spatial meter, unit m
    
    nMax = round(2 * Maxdepth / bin_res); % synethsis illumination kernel length
    
    if (length(t_in)<nMax)
        % Pad the virtual illumination kernel to required size
        t_out = padarray(t_in, round(nMax - length(t_in)), 'post');
    end
    
end


