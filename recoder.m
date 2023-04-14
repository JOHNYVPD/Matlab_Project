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

recorder =audiorecorder(Fs,nbits,ch);
disp('start')
recordblocking(recorder,Nseconds);
disp('end');
x=getaudiodata(recorder,datatype);
audiowrite('johny.wav',x,Fs);