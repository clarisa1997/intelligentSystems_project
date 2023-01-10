%% Configuration
clear
close all
clc

%Config variables
sequentialfs_rep = 5;
%param nfeatures for sequentialfs
nfeatures = 5;
EXTRACT_VALENCE = 1;
EXTRACT_AROUSAL = 1;

%% Features extraction
clean_dataset = load('data\clean_dataset.mat'); 
clean_dataset = clean_dataset.clean_dataset;

features = clean_dataset(:,3:end);
target_arousal = clean_dataset(:,1);
target_valence = clean_dataset(:,2);

cv = cvpartition(target_arousal, 'holdout', 0.3);
idxTraining = training(cv);
idxTesting = test(cv);

x_train = features(idxTraining, :);
y_train_arousal = target_arousal(idxTraining, :);
y_train_valence = target_valence(idxTraining, :);

x_test = features(idxTesting, :);
y_test_arousal = target_arousal(idxTesting, :);
y_test_valence = target_valence(idxTesting, :);



%% Features extraction for Arousal
if EXTRACT_AROUSAL == 1
    features_arousal = [zeros(1,54); 1:54]';
    %counter_feat_sel_arousal = zeros(54,1)';
    for i = 1:sequentialfs_rep
        fprintf("Iteration %i\n", i);
        %cvpartition discards rows of observations corresponding to missing values in group
        c = cvpartition(y_train_arousal, 'k', 10);
        option = statset('display','iter','useParallel',true);
        %Alla fine avrà selezionato 5 features
        inmodel_arousal = sequentialfs(@myfun, x_train, y_train_arousal, 'cv', c, 'opt', option, 'nFeatures', nfeatures);
        
        % Fetch useful indexes from result of latter sequentialfs
        %Per vedere quante volte in totale in tutte le iterazioni una
        %feature è stata selezionata
       for j = 1:54
            if inmodel_arousal(j) == 1
                features_arousal(j,1) = features_arousal(j,1) +1;
            end
        end
    end


    fprintf("\n");
    fprintf("*** AROUSAL: "); 
    fprintf("\n");

    disp(features_arousal);
    fprintf("Sorting features:\n");
    features_arousal = sortrows(features_arousal, 1, 'descend');
    disp(features_arousal);

    % Getting the 10 best arousal features
    arousal_best = features_arousal(1:10, 2);

    best_arousal_training.x_train = normalize(x_train(:, arousal_best));
    best_arousal_training.y_train = y_train_arousal';
    % Save struct
    save("data/training_arousal.mat", "best_arousal_training");

    best_arousal_testing.x_test = normalize(x_test(:, arousal_best));
    best_arousal_testing.y_test = y_test_arousal';
    save("data/testing_arousal.mat", "best_arousal_testing");
    fprintf("Arousal features saved\n");

end
%% Features extraction for valence

if EXTRACT_VALENCE == 1
    
    %Inizializzo due colonne e 54 righe, una con tutti 0 e l'altra con numeri da 1 a 54
    features_valence = [zeros(1,54); 1:54]';
    %counter_feat_sel_valence = zeros(54,1)';
    
    for i = 1:sequentialfs_rep
        fprintf("Iteration %i\n", i);
        
        c = cvpartition(y_train_valence, 'k', 10);
        option = statset('display','iter','useParallel',true);
        %inmodel contiene le 1 nelle festures selezionate
        inmodel_valence = sequentialfs(@myfun, x_train, y_train_valence, 'cv', c, 'opt', option, 'nFeatures', nfeatures);
        
        % Fetch useful indexes from result of latter sequentialfs
       for j = 1:54
            if inmodel_valence(j) == 1
                %riga j, jesima feature, incremento la colonna 1 che è il
                %contatore
                 features_valence(j, 1) = features_valence(j, 1) + 1;
            end
        end
    
    end
       
    fprintf("\n");
    fprintf("*** VALENCE: "); 
    fprintf("\n");
   
    disp(features_valence);
    fprintf("\n");
    
    disp(features_valence);
    fprintf("Sorting features:\n");
    features_valence = sortrows(features_valence, 1, 'descend');
    disp(features_valence);
   % Getting the 10 best valence features
    best_valence = features_valence(1:10, 2);

    best_valence_training.x_train = normalize(x_train(:, best_valence));
    best_valence_training.y_train = y_train_valence';
    % Save struct
    save("data/training_valence.mat", "best_valence_training");
    
    best_valence_testing.x_test = normalize(x_test(:, best_valence));
    best_valence_testing.y_test = y_test_valence';
    save("data/testing_valence.mat", "best_valence_testing");
    fprintf("valence features saved\n");
 
end


%% Save best-3 features arousal dataset for task 3.3
possible_values_arousal = unique(target_arousal);
possible_values_valence = unique(target_valence);
arousal_best3 = features_arousal(1:3, 2);
best3.x_train = normalize(x_train(:, arousal_best3));
best3.y_train = y_train_arousal';
best3.x_test = normalize(x_test(:, arousal_best3));
best3.y_test = y_test_arousal';
best3.best_features=arousal_best3;
best3.y_values= possible_values_arousal;
% Save struct
save("data/best3.mat", "best3");
fprintf("Best-3 arousal features saved\n");

%% Function for sequentialfs
function err = myfun(x_train, t_train, x_test, t_test)
    net = fitnet(60);
    net.trainParam.showWindow=0;
    % net.trainParam.showCommandLine=1;
    xx = x_train';
    tt = t_train';
    net = train(net, xx, tt);
    y=net(x_test'); 
    err = perform(net,t_test',y);
end


