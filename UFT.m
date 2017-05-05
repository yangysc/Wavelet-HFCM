% generate multi scale waves
clear
close all
clc
%% experimental data set
% dataset 1: enrollment
% enrollment = [13055, 13563, 13867, 14696, 15460, 15311, 15603, 15861, 16807, 16919, 16388, 15433, 15497, 15145, 15163, 15984, 16859, 18150, 18970, 19328, 19337, 18876]';
% dataset = enrollment;
pkg load io
% % dataset 2: TAIEX(need 85 % of data as train data
[NUM,TXT,RAW] = xlsread('2000_TAIEX.xlsx', 'clean_v1_2000');
dataset = NUM(1:end);


% dataset 3: sunspot
% dataset = importdata('sunspot.csv', ';');
% data = dataset(:, 2);
%dataset = data();
% dataset 4: Mackey-Glass chaos time series
% use fourth Runge-Kutta algorithm to create MG chaotic time series
% load mgdata.dat
%load MG_chaos
% dataset = mgdata(:, 2);
% dataset = dataset(124: 1123);
% synsitic data
% dataset = sin(1:100)';
% for i = 1 : length(dataset)
%     if dataset(i) ~= 0
%         dataset(i) = log(dataset(i));
%     end
% end
%dataset = dataset(1:1024);
dataset = (dataset - max(dataset)) ./ (max(dataset) - min(dataset));
pkg load ltfat
max_level = floor(log2(length(dataset)));
J = max_level - 1;  % level 5
%% use whole dataset(train + test) to conduct 'ufwt'
multi_data = ufwt(dataset, 'db1', J)';
%hold on
%for i = 1: 3
%		plot(multi_data(i, :))
%end
% save wavelet cofficicents into .mat file
save -6 multi_data.mat multi_data dataset

% use whole dataset(train + test) to conduct 'iufwt'
reversed_dataset = iufwt(multi_data', 'db1', J);
printf('err is %f\n', norm(dataset - reversed_dataset))
hold on
plot(reversed_dataset, 'r*')
plot(dataset)
title('use all dataset to ufwt and iufwt')
%
%%% use only train dataset to conduct 'ufwt' to prove
%%  if deleting the first 'Order' samples, the reverse tranform of next (Order * max_level)  samples will be inaccurate
%% since they need the information of the first 'Order' samples
%% split dataset into train_data and test_data
%ratio = 0.8;
%Order = 4;
%len_train_data  = floor(length(dataset) * ratio);
%train_data = dataset(1:len_train_data);
%% use only train dataset to conduct 'ufwt'
%multi_train_data = ufwt(train_data, 'db1', J)';
%% multi_train_data =  multi_data(:, Order:len_train_data);
%
%
%%% use only multi_train
%reveresed_train_data = iufwt(multi_train_data(:, Order + 1:end) ', 'db1', J)';
%figure()
%hold on
%plot(reveresed_train_data, 'r*--')
%plot(train_data(Order+1:end))
%legend('predicted(omit first Order samples', 'true')
%title('use train to ufwt and iufwt(but when iufwt omiting first Order samples')
%
%
%%% in next, show iufwt of ufwt of whole_data['train_data']  is not the same with
%%  iufwt of (ufwt of whole_data)['train_data'] 
%figure()
%reversed_only_train = iufwt(multi_train_data', 'db1', J)';
%reversed_train_ofWhole = iufwt(multi_data', 'db1', J)';
%reversed_train_ofWhole = reversed_train_ofWhole(1: len_train_data);
%hold on
%plot(reversed_only_train, 'r*--')
%plot(reversed_train_ofWhole)
%legend('predicted(only use train dataset', 'ufwt, ifwt using all then select first train')
%title('test if there is difference when has test data or not')
%
