%% Clean
clear
close all
clc
format compact


%% Load dataset

% Constant
PLOT_HIST_LOW = 1;
PLOT_HIST_MEDIUM = 1;
PLOT_HIST_HIGH = 1;

fuzzyData = load("data/best3.mat");
x_train = fuzzyData.best3.x_train;
y_train = fuzzyData.best3.y_train;
x_test = fuzzyData.best3.x_test;
y_test = fuzzyData.best3.y_test;

best_features = fuzzyData.best3.best_features;

y_values = fuzzyData.best3.y_values;

%Retrive the features name from the entire dataset
dataset = load("data/dataset.mat");

features_names = dataset.dataset.Properties.VariableNames(5:58);
fuzzyData_features_names = features_names(best_features);




%% Find universe of discourse 

x_24 = [x_train(:,1); x_test(:,1)];
x_27 = [x_train(:,2); x_test(:,2)];
x_35 = [x_train(:,3); x_test(:,3)];

x_24_tr = x_train(:,1);
x_27_tr = x_train(:,2);
x_35_tr = x_train(:,3);


max_24 = max(x_24);
max_27 = max(x_27);
max_35 = max(x_35);

min_24 = min(x_24);
min_27 = min(x_27);
min_35 = min(x_35);

% Plot Histogram of features for finding the empirical distribution

nbins = 15;
binWidth = 0.5;
y_lim = 30;

figure(1);
t = tiledlayout(1,3);
nexttile
histogram(x_24,'BinWidth',0.5);
x_24_name = features_names(:,1);
title('Histogram of feature24');
nexttile
histogram(x_27, 'BinWidth',0.5);
x_27_name = features_names(:,2);
title('Histogram of feature27');
nexttile
histogram(x_35, 'BinWidth',0.5);
x_35_name = features_names(:,3);
title('Histogram of feature35');

fprintf(" --- RANGES FOR UNIVERSE OF DISCOURSE ---\n");
fprintf("  Feature 24 -> Max:%f Min:%f\n", max_24, min_24);
fprintf("  Feature 27 -> Max:%f Min:%f\n", max_27, min_27);
fprintf("  Feature 35 -> Max:%f Min:%f\n", max_35, min_35);

%% Analysis of the features

% Check how many samples have a low/medium/high output that correspond to
% the different values of a particular feature

% Index of samples for each output
index1 = find(y_train == y_values(1));
index2 = find(y_train == y_values(2)); 
index3 = find(y_train == y_values(3));
index4 = find(y_train == y_values(4));
index5 = find(y_train == y_values(5));
index6 = find(y_train == y_values(6));
index7 = find(y_train == y_values(7));

y_low = [index1 index2];
y_medium = [index3 index4 index5];
y_high = [index6 index7];

figure(4)
t = tiledlayout(1,3);
nexttile

%Feature 27
histogram(x_train(y_low,1), 'BinWidth', binWidth);
yline(y_lim, '--r');
title('Feature 24 low');
nexttile
histogram(x_train(y_medium,1), 'BinWidth', binWidth);
yline(y_lim, '--r');
title('Feature 24 medium');
nexttile
histogram(x_train(y_high,1), 'BinWidth', binWidth);
yline(y_lim, '--r');
title('Feature 24 high');


%Feature 11
figure(5)
t = tiledlayout(1,3);
nexttile
histogram(x_train(y_low,2), 'BinWidth', binWidth);
yline(y_lim, '--r');
title('Feature 27 low');
nexttile
histogram(x_train(y_medium,2), 'BinWidth', binWidth);
yline(y_lim, '--r');
title('Feature 27 medium');
nexttile
histogram(x_train(y_high,2), 'BinWidth', binWidth);
yline(y_lim, '--r');
title('Feature 27 high');

%Feature 13
figure(6)
t = tiledlayout(1,3);
nexttile
histogram(x_train(y_low,3), 'BinWidth', binWidth);
yline(y_lim, '--r');
title('Feature 35 low');
nexttile
histogram(x_train(y_medium,3), 'BinWidth', binWidth);
yline(y_lim, '--r');
title('Feature 35 medium');
nexttile
histogram(x_train(y_high,3), 'BinWidth', binWidth);
yline(y_lim, '--r');
title('Feature 35 high');


%Low
%if PLOT_HIST_LOW==1
%    figure(4);
%    histogram(x_train(y_low,1));
%    title("Low Outputs For Feature 24");
%    figure(5);
%    histogram(x_train(y_low,2));
%    title("Low Outputs For Feature 27");
%    figure(6);
%    histogram(x_train(y_low,3));
%    title("Low Outputs For Feature 35");

  
%end

%Medium
%if PLOT_HIST_MEDIUM==1
%   figure(7);
%    histogram(x_train(y_medium,1));
%    title("Medium Outputs For Feature 24");
%    figure(8);
%    histogram(x_train(y_medium,2));
%    title("Medium Outputs For Feature 27");
%    figure(9);
%    histogram(x_train(y_medium,3));
%    title("Medium Outputs For Feature 35");
%end

%High
%if PLOT_HIST_HIGH==1
%    figure(10);
%    histogram(x_train(y_high,1));
%    title("High Outputs For Feature 24");
%    figure(27);
%    histogram(x_train(y_high,2));
%    title("High Outputs For Feature 27");
%    figure(35);
%    histogram(x_train(y_high,3));
%    title("High Outputs For Feature 35");
%end

% Plot Scatterplots between pairs of features to find some correlations
figure(7)
%t = tiledlayout(1,3);
%nexttile
scatter(x_24, x_27);
title('Scatterplot of feature 24 and feature 27');
%nexttile
figure(8)
scatter(x_24, x_35);
title('Scatterplot of feature 24 and feature 35');
%nexttile
figure(9)
scatter(x_27, x_35);
title('Scatterplot of feature 27 and feature 35');

