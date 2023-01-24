%% Evaluation of FIS
clc;
clear;
close all;

fis = readfis('mamdaniFuzzy');

fuzzyData = load("data/best3.mat");

x_test = fuzzyData.best3.x_test;
y_test = fuzzyData.best3.y_test;

%output = evalfis(fis,x_test);
%plotregression(y_test, output);
output=evalfis(fis, [1 1 2]);

