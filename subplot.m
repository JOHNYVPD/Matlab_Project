clc;
clear all;
close all;




%% Record Audio Data
%%


Fs = 16000 ; % sampling frequency
ch = 1 ; % number of channels (Mono)
data_type = 'uint8' ; % Data type
nbits = 16 ; % number of bits
Nseconds = 10 ; % duration of the record


%%


recorder_1 = audioread('johny.wav') 


%%


recorder_2 = audioread('test.wav')


%%


audio_data_1 = recorder_1.' ;
audio_data_2 = recorder_2.' ;

% Define Time Axis
dt = 1/Fs ;
t = 0 : dt : 10-dt ;



%% FFT of Audio Data
%%


N = 128 ; % FFT Point Number
df = Fs/N ;

% Define f axis for N point FFT
f = -Fs/2 : df : Fs/2-df ;

fft_audio_data_1 = fft(audio_data_1,N) ; % FFT of Audio Data 1

fft_audio_data_2 = fft(audio_data_2,N) ; % FFT of Audio Data 2



%% LOW PASS IIR FILTER
%%


Fc = 4000 ; % Cutt-Off Frequency
Ts = 1/Fs ; % sampling period

% Filter Pre-Wraped Frequency Calculation
Wd = 2*pi*Fc ; % Digital Frequency
Wa = (2/Ts)*tan((Wd*Ts)/2) ; %pre-Wraped Frequency

% Analog Filter Coefficients H(s) = 1/(1+s)
num = 1 ; % Numerator Coefficients
den = [1 1] ; % Denominator Coefficients

% Filter Transformation from Low Pass to Low Pass 
[A, B] = lp2lp(num, den, Fs) ;
[a, b] = bilinear(A, B, Fs) ;

% Frequency Response
[hz, fz] = freqz(a, b, N, Fs) ;
phi = 180*unwrap(angle(hz))/pi ;




%% Filtering the Audio Data
%%


% Filtering Audio Data 1
filtered_audio_data_1 = filter(a,b,audio_data_1) ;

% N FFT of Filtered Audio Data 1
fft_filtered_audio_data_1 = fft(filtered_audio_data_1,N) ;


%%


% Filtering Audio Data 2
filtered_audio_data_2 = filter(a,b,audio_data_2) ;

% N FFT of Filtered Audio Data 2
fft_filtered_audio_data_2 = fft(filtered_audio_data_2,N) ;




%% Ploting IIR Low Pass Filter Output
%%


figure(1)
subplot(221)
stem(t,filtered_audio_data_1,'r')
title(' Filtered Data 1 (Time Domain) ')
xlabel('Time (sec) ')
ylabel('  Magnitude  ')

subplot(222)
stem(f,fftshift(abs(fft_filtered_audio_data_1)),'r')
title(' Low pass Filtered Data 1 (Frequency Domain) ')

xlabel('Frequency (Hz) ')
ylabel(' | Magnitude | ')

subplot(223)
stem(t,filtered_audio_data_2,'y')
title(' Filtered Data 2 (Time Domain) ')

xlabel('Time (sec) ')
ylabel('  Magnitude  ')

subplot(224)
stem(f,fftshift(abs(fft_filtered_audio_data_2)),'y')
title(' Low pass Filtered Data 2 (Frequency Domain) ')
xlabel('Frequency (Hz) ')
ylabel(' | Magnitude | ')

%% Carrier Signal
%%


% Carrier Frequency
fc_1 = 121000 ;
fc_2 = fc_1*6 ;

% Define F axis
L = length(audio_data_1) ;
F_axis = fc_1*20 ;
fs = F_axis ;
ts = 1/fs ;
F = (0 : 1/L : 1-(1/L))*fs - (fs/2) ;

% define n axis
A = length(audio_data_1)/2 ;
n = -A*ts : ts : A*ts-ts ;

% Carrier
Carrier_1 = cos(2*pi*fc_1*n) ; % Carrier signal 1

Carrier_2 = cos(2*pi*fc_2*n) ; % Carrier signal 2




%% FFT of CARRIER SIGNAL
%%


% FFT of Carrier 1
C_1 = fft(Carrier_1) ;


%%


% FFT of Carrier 2
C_2 = fft(Carrier_2) ;



%% MODULATION
%%


% Modulation of Data 1
modulated_data_1 = filtered_audio_data_1.*Carrier_1


%%


% Modulation of Data 2
modulated_data_2 = filtered_audio_data_2.*Carrier_2




%% FFT SPECTRUM OF MODULATED SIGNAL
%%


% FFT of Modulated Data 1
fft_modulated_data_1 = fft(modulated_data_1) 


%%


% FFT of Modulated Data 2
fft_modulated_data_2 = fft(modulated_data_2) 




%% Ploting Modulated Signal
%%


figure(2)
subplot(221)
stem(t,modulated_data_1)
title(' Modulated Audio Data 1 (Time Domain) ')
xlabel( 'Time (sec) ')
ylabel('  Magnitude  ')
grid on

subplot(222)
stem(F,fftshift(abs(fft_modulated_data_1)))
title(' Modulated Audio Data 1 (Frequency Domain) ')
xlabel('Frequency (Hz) ')
ylabel(' | Magnitude | ')
grid on

subplot(223)
stem(t,modulated_data_2)
title(' Modulated Audio Data 2 (Time Domain) ')
xlabel( 'Time (sec) ')
ylabel('  Magnitude  ')
grid on

subplot(224)
stem(F,fftshift(abs(fft_modulated_data_2)))
title(' Modulated Audio Data 2 (Frequency Domain) ')
xlabel('Frequency (Hz) ')
ylabel(' | Magnitude | ')
grid on


%% FREQUENCY DIVISION MULTIPLEXING
%%


FDM_Audio_Data = fft_modulated_data_1 + fft_modulated_data_2


%%


ifft_FDM_Audio_Data = ifft(FDM_Audio_Data)




%% Plotting FDM Audio Data
%%

figure(3)
subplot(221)
stem(F,fftshift(abs(FDM_Audio_Data)))
title(' Modulated Audio Data (Frequency Domain) ')
xlabel('Frequency (Hz) ')
ylabel(' | Magnitude | ')
grid on

subplot(222)
stem(t,ifft_FDM_Audio_Data )
title(' Modulated Audio Data (Time Domain) ')
xlabel(' Time (sec) ')
ylabel('  Magnitude  ')
grid on


%% Ideal Bandpass filter using find function
%% BANDPASS FILTER FOR DATA 1


Limit = max(abs(fft_modulated_data_1))*0.15
fft_modulated_data_1_zeros = find(fft_modulated_data_1 > Limit) ;

U = zeros(1,L) ;
U(fft_modulated_data_1_zeros) = 1 ;
 
first_half_1 = zeros(1,L) ;
first_half_1  = U(1 : L/2) ;
first_half_1_zeros = find(first_half_1) ;
ideal_BP_filter_1 = zeros(1, L) ;
ideal_BP_filter_1( min(first_half_1_zeros) : max(first_half_1_zeros) ) = 1 ;

second_half_1 = zeros(1,L) ;
second_half_1(L/2 : L)  = U(L/2 : L) ;
second_half_1_zeros = find(second_half_1) ;
ideal_BP_filter_1( min(second_half_1_zeros) : max(second_half_1_zeros) ) = 1 ;


%% BANDPASS FILTER FOR DATA 2

Limit = max(abs(fft_modulated_data_2))*0.15
fft_modulated_data_2_zeros = find(fft_modulated_data_2 > Limit) ;

U = zeros(1,L) ;
U(fft_modulated_data_2_zeros) = 1 ;
 
first_half_2 = zeros(1,L) ;
first_half_2  = U(1 : L/2) ;
first_half_2_zeros = find(first_half_2) ;
ideal_BP_filter_2 = zeros(1, L) ;
ideal_BP_filter_2( min(first_half_2_zeros) : max(first_half_2_zeros) ) = 1 ;

second_half_2 = zeros(1,L) ;
second_half_2(L/2 : L)  = U(L/2 : L) ;
second_half_2_zeros = find(second_half_2) ;
ideal_BP_filter_2( min(second_half_2_zeros) : max(second_half_2_zeros) ) = 1 ;


%% Applying Ideal Band-Pass Filter
%%


% Applying Ideal Band-Pass Filer_1
Extracted_Data_1 = ideal_BP_filter_1.*FDM_Audio_Data

% Applying Ideal Band-Pass Filer_1
Extracted_Data_2 = ideal_BP_filter_2.*FDM_Audio_Data


%% Time Domain Output of Ideal Band-Pass Filter
%%


% Output from Bandpass Filter 1 (Time Domain)
IFFT_Extracted_Audio_Data_1 = ifft(Extracted_Data_1)

% Output from Bandpass Filter 1 (Time Domain)
IFFT_Extracted_Audio_Data_2 = ifft(Extracted_Data_2)


%% Ploting Ideal Band-Pass Filter Output
%%


figure(4)
subplot(221)
stem(t,IFFT_Extracted_Audio_Data_1)
title(' Bandpass Filter Output 1 (Time Domain) ')
xlabel( 'Time (sec) ')
ylabel('  Magnitude  ')
grid on

subplot(222)
stem(F,fftshift(abs(Extracted_Data_1)))
title(' Bandpass Filter Output 1 (Frequency Domain) ')
xlabel('Frequency (Hz) ')
ylabel(' | Magnitude | ')
grid on

subplot(223)
stem(t,IFFT_Extracted_Audio_Data_2)
title(' Bandpass Filter Output 2 (Time Domain) ')
xlabel( 'Time (sec) ')
ylabel('  Magnitude  ')
grid on

subplot(224)
stem(F,fftshift(abs(Extracted_Data_2)))
title(' Bandpass Filter Output 2 (Frequency Domain) ')
xlabel('Frequency (Hz) ')
ylabel(' | Magnitude | ')
grid on


%% DEMODULATION
%%


% Demodulation of Data 1
Demodulated_Audio_Data_1 = IFFT_Extracted_Audio_Data_1.*Carrier_1

% Demodulation of Data 2
Demodulated_Audio_Data_2 = IFFT_Extracted_Audio_Data_2.*Carrier_2




%% FFT of Demodulated Data
%%

% FFT of Demodulated Data 1
FFT_Demodulated_Audio_Data_1 = fft(Demodulated_Audio_Data_1)

% FFT of Demodulated Data 2
FFT_Demodulated_Audio_Data_2 = fft(Demodulated_Audio_Data_2)




%% Ploting Demodulated Data
%%


figure(5)
subplot(221)
stem(F,fftshift(abs(FFT_Demodulated_Audio_Data_1)))
title(' Demodulated Audio Data 1 ( Frequency Domain) ')
xlabel(' Frequency (Hz) ')
ylabel(' | Magnitude | ')
grid on

subplot(222)
stem(F,fftshift(abs(FFT_Demodulated_Audio_Data_2)))
title(' Demodulated Audio Data 2 ( Frequency Domain) ')
xlabel(' Frequency (Hz) ')
ylabel(' | Magnitude | ')
grid on

subplot(223)
stem(t,Demodulated_Audio_Data_1)
title(' Demodulated Audio Data 1 ( Time Domain) ')
xlabel(' Time (sec) ')
ylabel(' Magnitude ')
grid on

subplot(224)
stem(t,Demodulated_Audio_Data_2)
title(' Demodulated Audio Data 2 ( Time Domain) ')
xlabel(' Time (sec) ')
ylabel(' Magnitude ')
grid on



%% LOW PASS FIR FILTER
%%


TW = 1000 ; % Transition Width
PBE = 3500; % Passband Edge Frequency
ntw = TW/Fs ; % Normalized Transition Width
M = round(5.5/ntw) ; % Filter Order for Blackman Filter
Fc = PBE + TW/2; % Corner Frequency
Wc = 2*pi*(Fc/Fs) ; % Converting Wc to 2*pi range

a = fir1(M,Wc/pi,'low',blackman(M+1));
N = 80000 ; % Number of Points for Calculating Frequency Response
[h ff] = freqz(a,1,N,Fs)

% Converting Filter Length
H = h.' ;
LPF_H = zeros(1,L) 
LPF_H( 1 : L/2 ) = H( 1 : length(H) )



%% FIR LOW PASS FILTER OUTPUT (FREQUENCY DOMAIN)
%%


% FIR LOW PASS FILTER OUTPUT 1 (Frequency Domain)
FIR_LPF_Output_1 = FFT_Demodulated_Audio_Data_1.*LPF_H

% FIR LOW PASS FILTER OUTPUT 2 (Frequency Domain)
FIR_LPF_Output_2 = FFT_Demodulated_Audio_Data_2.*LPF_H




%% FIR LOW PASS FILTER OUTPUT (TIME DOMAIN)
%%


% FIR LOW PASS FILTER OUTPUT 1 (Time Domain)
Audio_Data_1_Output = ifft(FIR_LPF_Output_1)

% FIR LOW PASS FILTER OUTPUT 2 (Time Domain)
Audio_Data_2_Output = ifft(FIR_LPF_Output_2)




%% Ploting FIR LOW PASS FILTER OUTPUT
%%


figure(6)
subplot(221)
stem(t,Audio_Data_1_Output)
title(' FIR LOW PASS FILTER OUTPUT 1 (Time Domain) ')
xlabel(' Time (sec) ')
ylabel(' Magnitude ')
grid on

subplot(222)
stem(F,fftshift(abs(FIR_LPF_Output_1)))
title(' FIR LOW PASS FILTER OUTPUT 2 (Frequency Domain) ')
xlabel(' Frequency (Hz) ')
ylabel(' | Magnitude | ')
grid on

subplot(223)
stem(t,Audio_Data_2_Output)
title(' FIR LOW PASS FILTER OUTPUT 2 (Time Domain) ')
xlabel(' Time (sec) ')
ylabel(' Magnitude ')
grid on

subplot(224)
stem(F,fftshift(abs(FIR_LPF_Output_2)))
title(' FIR LOW PASS FILTER OUTPUT 2 (Frequency Domain) ')
xlabel(' Frequency (Hz) ')
ylabel(' | Magnitude | ')
grid on

%% AMPLIFIER GAIN
%% OUTPUT DATA 1 GAIN


Audio_Data_1_Part_1 = zeros(1,L)
Data_1_Output_Part_1 = zeros(1,L)
Audio_Data_1_Part_1( 1 : L/4 ) = audio_data_1( 1 : L/4 )
Data_1_Output_Part_1( 1 : L/4 ) = Audio_Data_1_Output( 1 : L/4 )
Gain_1 = max(abs(Audio_Data_1_Part_1))/max(abs(Data_1_Output_Part_1))
Amplified_Output_1 = Gain_1 * Audio_Data_1_Part_1




Audio_Data_1_Part_2 = zeros(1,L)
Data_1_Output_Part_2 = zeros(1,L)
Audio_Data_1_Part_2( L/4 : L/2 ) = audio_data_1( L/4 : L/2 )
Data_1_Output_Part_2( L/4 : L/2 ) = Audio_Data_1_Output( L/4 : L/2 )
G_2 = max(abs(Audio_Data_1_Part_2))/max(abs(Data_1_Output_Part_2))
Amplified_Output_2 = G_2 * Audio_Data_1_Part_2




Audio_Data_1_Part_3 = zeros(1,L)
Data_1_Output_Part_3 = zeros(1,L)
Audio_Data_1_Part_3( L/2 : 3*L/4) = audio_data_1( L/2 : 3*L/4)
Data_1_Output_Part_3( L/2 : 3*L/4 ) = Audio_Data_1_Output( L/2 : 3*L/4 )
Gain_3 = max(abs(Audio_Data_1_Part_3))/max(abs(Data_1_Output_Part_3))
Amplified_Output_3 = Gain_3 * Audio_Data_1_Part_3




Audio_Data_1_Part_4 = zeros(1,L)
Data_1_Output_Part_4 = zeros(1,L)
Audio_Data_1_Part_4( 3*L/4 : L ) = audio_data_1( 3*L/4 : L )
Data_1_Output_Part_4( 3*L/4 : L ) = Audio_Data_1_Output( 3*L/4 : L )
Gain_4 = max(abs(Audio_Data_1_Part_4))/max(abs(Data_1_Output_Part_4))
Amplified_Output_4 = Gain_4 * Audio_Data_1_Part_4


Data_1_Output = Amplified_Output_1 + Amplified_Output_2 + Amplified_Output_3 + Amplified_Output_4




%% OUTPUT DATA 2 GAIN


Audio_Data_2_Part_1 = zeros(1,L)
Data_2_Output_Part_1 = zeros(1,L)
Audio_Data_2_Part_1( 1 : L/4 ) = audio_data_2( 1 : L/4 )
Data_2_Output_Part_1( 1 : L/4 ) = Audio_Data_2_Output( 1 : L/4 )
Gain_1 = max(abs(Audio_Data_2_Part_1))/max(abs(Data_2_Output_Part_1))
Amplified_Output_1 = Gain_1 * Audio_Data_2_Part_1




Audio_Data_2_Part_2 = zeros(1,L)
Data_2_Output_Part_2 = zeros(1,L)
Audio_Data_2_Part_2( L/4 : L/2 ) = audio_data_2( L/4 : L/2 )
Data_2_Output_Part_2( L/4 : L/2 ) = Audio_Data_2_Output( L/4 : L/2 )
G_2 = max(abs(Audio_Data_2_Part_2))/max(abs(Data_2_Output_Part_2))
Amplified_Output_2 = G_2 * Audio_Data_2_Part_2




Audio_Data_2_Part_3 = zeros(1,L)
Data_2_Output_Part_3 = zeros(1,L)
Audio_Data_2_Part_3( L/2 : 3*L/4) = audio_data_2( L/2 : 3*L/4)
Data_2_Output_Part_3( L/2 : 3*L/4 ) = Audio_Data_2_Output( L/2 : 3*L/4 )
Gain_3 = max(abs(Audio_Data_2_Part_3))/max(abs(Data_2_Output_Part_3))
Amplified_Output_3 = Gain_3 * Audio_Data_2_Part_3




Audio_Data_2_Part_4 = zeros(1,L)
Data_2_Output_Part_4 = zeros(1,L)
Audio_Data_2_Part_4( 3*L/4 : L ) = audio_data_2( 3*L/4 : L )
Data_2_Output_Part_4( 3*L/4 : L ) = Audio_Data_2_Output( 3*L/4 : L )
Gain_4 = max(abs(Audio_Data_2_Part_4))/max(abs(Data_2_Output_Part_4))
Amplified_Output_4 = Gain_4 * Audio_Data_2_Part_4


Data_2_Output = Amplified_Output_1 + Amplified_Output_2 + Amplified_Output_3 + Amplified_Output_4

%%


fft_Data_1_Output = ifft(Data_1_Output,128)
fft_Data_2_Output = ifft(Data_2_Output,128)




%% Ploting Output
%%


figure(37)
stem(f,fftshift(abs(fft_Data_1_Output))) ;
title('Output Data 1 (Frequency Domain)')
xlabel(' Frequency (Hz) ')
ylabel(' | Magnitude | ')
grid on

figure(38)
stem(t,Data_1_Output)
title('Output Data 1 (Time Domain)')
xlabel(' Time (sec) ')
ylabel(' Magnitude ')
grid on

figure(39)
stem(f,fftshift(abs(fft_Data_2_Output))) ;
title('Output Data 2 (Frequency Domain)')
xlabel(' Frequency (Hz) ')
ylabel(' | Magnitude | ')
grid on

figure(40)
stem(t,Data_2_Output)
title('Output Data 2 (Time Domain)')
xlabel(' Time (sec) ')
ylabel(' Magnitude ')
grid on



%% Creating Output Audio File
%%

% Output Audio File 1
audiowrite( 'Output_johny.wav',Data_1_Output,16000 )

%%

% Output Audio File 2
audiowrite( 'Output_test.wav',Data_2_Output,16000 )




