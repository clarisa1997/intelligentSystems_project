%% Clean
clear
close all
clc
format compact

%% Load the features

%Load results obtained from sequential feature selection

test_arousal = load('data/testing_arousal.mat'); 
train_arousal = load('data/training_arousal.mat');
fprintf("Arousal features loaded\n");

test_valence = load('data/testing_valence.mat');
train_valence = load('data/training_valence.mat');
fprintf("Valence features loaded\n");

%Data for training
x_train_arousal = train_arousal.best_arousal_training.x_train';
t_train_arousal = train_arousal.best_arousal_training.y_train'.';

x_train_valence = train_valence.best_valance_training.x_train';
t_train_valence = train_valence.best_valance_training.y_train'.';

%Data for final test of the network to assess performance
x_test_arousal = test_arousal.best_arousal_testing.x_test';
t_test_arousal = test_arousal.best_arousal_testing.y_test'.';

x_test_valence = test_valence.best_valance_testing.x_test';
t_test_valence = test_valence.best_valance_testing.y_test'.';


MLP_AROUSAL = 0;
MLP_VALENCE = 0;
RBFN_AROUSAL = 0;
RBFN_VALENCE = 0;
TESTING_AROUSAL = 1;
TESTING_VALANCE = 0;

%% Training MLP FOR AROUSAL

if MLP_AROUSAL == 1
    % Optimal Neural Network Architecture found for arousal
    mlp_net_arousal = fitnet(45);
    mlp_net_arousal.divideParam.trainRatio = 0.7;
    mlp_net_arousal.divideParam.testRatio = 0.1; %Just to see the difference with the other test set
    mlp_net_arousal.divideParam.valRatio = 0.2;
    mlp_net_arousal.trainParam.showWindow = 1;
    mlp_net_arousal.trainParam.showCommandLine = 1;
    mlp_net_arousal.trainParam.lr = 0.1; 
    mlp_net_arousal.trainParam.epochs = 100;
    mlp_net_arousal.trainParam.max_fail = 10;
    
    [mlp_net_arousal, tr_arousal] = train(mlp_net_arousal, x_train_arousal, t_train_arousal);
    view(mlp_net_arousal);
    figure(1);
    plotperform(tr_arousal);
    y_test_arousal = mlp_net_arousal(x_test_arousal);
    figure(2);
    plotregression(t_test_arousal, y_test_arousal, ['Final test arousal 45 neurons: ']);

elseif TESTING_AROUSAL == 1

% Traces of other experiments
    max_neurons_1 = 120;
    for i=5:5:max_neurons_1    
        mlp_net_arousal = fitnet(i);
        mlp_net_arousal.divideParam.trainRatio = 0.7;
        mlp_net_arousal.divideParam.testRatio = 0.1; 
        mlp_net_arousal.divideParam.valRatio = 0.2;
        mlp_net_arousal.trainParam.showWindow = 0;
        mlp_net_arousal.trainParam.showCommandLine = 1;
        mlp_net_arousal.trainParam.lr = 0.05; 
        mlp_net_arousal.trainParam.epochs = 100;
        mlp_net_arousal.trainParam.max_fail = 10;
        [mlp_net_arousal, tr_arousal] = train(mlp_net_arousal, x_train_arousal, t_train_arousal);
        %view(mlp_net_arousal);
        figure(1);
        plotperform(tr_arousal);
        path_arousal_perf = "results/images/mpl_arousal_training/perf_arousal_" + i + "_neurons";
        saveas(figure(1),path_arousal_perf);
        y_test_arousal = mlp_net_arousal(x_test_arousal);
        figure(2);
        plotregression(t_test_arousal, y_test_arousal, ['Final test arousal: ' string(i)]);
        path_arousal_reg = "results/images/mpl_arousal_training/reg_arousal_" + i + "_neurons";
        saveas(figure(2),path_arousal_reg);
    end
end

%%  Train MLP for valence

if MLP_VALENCE == 1

    % Optimal Neural Network Architecture found for valence
    mlp_net_valence = fitnet(80);
    mlp_net_valence.divideParam.trainRatio = 0.7; 
    mlp_net_valence.divideParam.valRatio = 0.2; 
    mlp_net_valence.divideParam.testRatio = 0.1;
    mlp_net_valence.trainParam.showWindow = 1;
    mlp_net_valence.trainParam.showCommandLine = 1;
    mlp_net_valence.trainParam.lr = 0.1; 
    mlp_net_valence.trainParam.epochs = 100;
    mlp_net_valence.trainParam.max_fail = 15;
    [mlp_net_valence, tr_valence] = train(mlp_net_valence, x_train_valence, t_train_valence);
    view(mlp_net_valence);
    figure(1);
    plotperform(tr_valence);
    y_test_valence = mlp_net_valence(x_test_valence);
    figure(2);
    plotregression(t_test_valence, y_test_valence, ['Final test valence: 80 hidden Neurons']);

elseif TESTING_VALANCE == 1

% Traces of other experiments
    max_neurons = 100;
    for i=5:5:max_neurons
        mlp_net_valence = fitnet(i);
        mlp_net_valence.divideParam.trainRatio = 0.8; 
        mlp_net_valence.divideParam.valRatio = 0.2; 
        mlp_net_valence.divideParam.testRatio = 0;
        mlp_net_valence.trainParam.showWindow = 0;
        mlp_net_valence.trainParam.showCommandLine = 1;
        mlp_net_valence.trainParam.lr = 0.1; 
        mlp_net_valence.trainParam.epochs = 100;
        mlp_net_valence.trainParam.max_fail = 15;
        [mlp_net_valence, tr_valence] = train(mlp_net_valence, x_train_valence, t_train_valence);
        %view(mlp_net_valence);
        figure(1);
        plotperform(tr_valence);
        path_valance_perf = "results/images/mpl_valance_training/perf_valance_" + i + "_neurons";
        saveas(figure(1),path_valance_perf);
        y_test_valence = mlp_net_valence(x_test_valence);
        figure(2);
        plotregression(t_test_valence, y_test_valence, ['Final test valence: ' string(i)]);
        path_valance_reg = "results/images/mpl_valance_training/reg_valance_" + i + "_neurons";
        saveas(figure(2),path_valance_reg);
    end
end

%% Part with RBF training for arousal

if RBFN_AROUSAL == 1
    %Parameters for training
    spread_ar = 1.07;
    goal_ar = 0;
    K_ar = 1200;
    Ki_ar = 100; %in order to speed up the training instead of the default 50
    
    rbf_arousal = newrb(x_train_arousal,t_train_arousal,goal_ar,spread_ar,K_ar,Ki_ar);
    
    % Test RBF
    y_test_arousal = rbf_arousal(x_test_arousal);
    plotregression(t_test_arousal, y_test_arousal, 'Final test arousal with RBF');
end

%% Part with RBF training for valence

if RBFN_VALENCE == 1
    %Parameters for training
    spread_va = 0.7;
    goal_va = 0;
    K_va = 1200;
    Ki_va = 100; %in order to speed up the training instead of the default 50
    
    rbf_valence = newrb(x_train_valence,t_train_valence,goal_va,spread_va, K_va, Ki_va);
    
    % Test RBF
    y_test_valence = rbf_valence(x_test_valence);
    plotregression(t_test_valence, y_test_valence, 'Final test valence with RBF');
end


