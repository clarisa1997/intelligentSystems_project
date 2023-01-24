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
RBFN_AROUSAL = 1;
RBFN_VALENCE = 0;
TESTING_AROUSAL = 0;
TESTING_VALANCE = 0;

%% Training MLP FOR AROUSAL

if MLP_AROUSAL == 1
    % Optimal Neural Network Architecture found for arousal
    mlp_net_arousal = fitnet(40);
    mlp_net_arousal.divideParam.trainRatio = 0.7;
    mlp_net_arousal.divideParam.testRatio = 0.1; 
    mlp_net_arousal.divideParam.valRatio = 0.2;
    mlp_net_arousal.trainParam.showWindow = 1;
    mlp_net_arousal.trainParam.showCommandLine = 1;
    mlp_net_arousal.trainParam.lr = 0.05; 
    mlp_net_arousal.trainParam.epochs = 100;
    mlp_net_arousal.trainParam.max_fail = 15;
    
    [mlp_net_arousal, tr_arousal] = train(mlp_net_arousal, x_train_arousal, t_train_arousal);
    view(mlp_net_arousal);
    figure(1);
    plotperform(tr_arousal);
    y_test_arousal = mlp_net_arousal(x_test_arousal);
    figure(2);
    plotregression(t_test_arousal, y_test_arousal, ['Final test arousal 40 neurons: ']);

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
    mlp_net_valence = fitnet(50);
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
    plotregression(t_test_valence, y_test_valence, ['Final test valence: 50 hidden Neurons']);

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
    spread = 1;
    goal = 0.02;
    max_neurons = 140;
    neurons_to_add = 20; 
    
    rbf_arousal = newrb(x_train_arousal,t_train_arousal,goal,spread, max_neurons, neurons_to_add);
    
    view(rbf_arousal);

    % Test RBF
    output_test_arousal = rbf_arousal(x_test_arousal);

    plotregression(t_test_arousal, output_test_arousal, 'Final test arousal with RBF');
    
    figure(1);
    plot(x_test_arousal, t_test_arousal,'r');

    figure(2);
    plot(x_test_arousal, output_test_arousal, 'b--');

end

%% Part with RBF training for valence

if RBFN_VALENCE == 1
    %Parameters for training
    spread = 1;
    goal = 0.02;
    max_neurons = 160;
    neurons_to_add = 20; 
    
    rbf_valence = newrb(x_train_valence,t_train_valence,goal,spread, max_neurons, neurons_to_add);

    view(rbf_valence);
    
    % Test RBF
    output_test_valence = rbf_valence(x_test_valence);

    plotregression(t_test_valence, output_test_valence, 'Final test valence with RBF');

    figure(1);
    plot(x_test_valence, t_test_valence,'r');

    figure(2);
    plot(x_test_valence, output_test_valence, 'b--');
end


