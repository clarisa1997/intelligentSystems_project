%% Configuration
clear
close all
clc

%% Preparation of Data

outliers_removal_method = 'median';
% Removal of non numerical values
dataset = load('data\dataset.mat'); 
dataset = table2array(dataset.dataset);
inf_val = isinf(dataset);
[rows_inf, col_inf] = find(inf_val == 1); %indici della cella Inf
dataset(rows_inf,:) = []; %elimino tutti i valori della riga che ha un Inf

% Removal of outliers
dataset = dataset(:, 3:end); %Adesso il dataset sono solo le features senza gli id di soggetti e video
clean_dataset = rmoutliers(dataset, outliers_removal_method);
%save("data/clean_dataset.mat", "clean_dataset");
[final_rows, ~] = size(clean_dataset); %size mi da sia righe che colonne


%% Data Balancing

%getting arousal and valence levels
arousal_level  = clean_dataset(:,1);
valence_level = clean_dataset(:,2);

% number of samples for each level of arousal and valence
sample_arousal = groupcounts(arousal_level);
sample_valence = groupcounts(valence_level);


possible_values_arousal = unique(arousal_level);
possible_values_valence = unique(valence_level);


% plot the graph
figure("Name", "Sample for arousal before balancing");
bar(sample_arousal);
title("Sample for arousal before balancing");

fprintf("Data are unbalanced\n");

[~, min_arousal] = min(sample_arousal);
[~, max_arousal] = max(sample_arousal);

% plot the graph
figure("Name", "Sample for valence before balancing");
bar(sample_valence);
title("Sample for valence before balancing");

fprintf("Data are unbalanced\n");

%Indice del valore min e max
[~, min_valence] = min(sample_valence);
[~, max_valence] = max(sample_valence);

augmentation_factors = [0 0];

debug = clean_dataset;

% Oppure possible:values_arousal tanto sono gli stessi valori per entrambi
possible_values = possible_values_valence;
rep = 80;
row_to_check = final_rows;

for k = 1:rep
    for i = 1:row_to_check
        %Se questa riga appartiene alla classe (di arousal) che ha meno
        %campioni, però valence non appartine al classe col massimo e
        %viceversa allora faccio augmentation su quello minimo
        if (clean_dataset(i,1)==possible_values(min_arousal) && clean_dataset(i,2)~=possible_values(max_valence)) || (clean_dataset(i,1)~=possible_values(max_arousal) && clean_dataset(i,2)==possible_values(min_valence))
            % Selection of i-th row
            selected_row = clean_dataset(i,:);
            % Augmentation of the i-th row
            row_to_add = selected_row;
            % Selection of the augmentation factor
            augmentation_factors(1) = 0.95+(0.04)*rand;
            augmentation_factors(2) = 1.01+(0.04)*rand;
            j = round(0.51+(1.98)*rand);
            % Augmentation, le prime 2 colonne le stesse (livelli), le
            % altre le cambio attraverso i fattori di augmentation
            row_to_add(3:end) = selected_row(3:end).*augmentation_factors(j); 
            % Addition of the new sample, obtained through augmentation, to
            % the dataset
            clean_dataset = [clean_dataset; row_to_add];
        end
        
        if((clean_dataset(i,1)==possible_values(max_arousal) && clean_dataset(i,2)~=possible_values(min_valence)) || (clean_dataset(i,2)==possible_values(max_valence) && clean_dataset(i,1)~=possible_values(min_arousal)))
            clean_dataset(i,:)=[];
            %fprintf(" I am removing the row %i\n",i);
        end
    end
    samples_arousal = groupcounts(clean_dataset(:,1));
    samples_valence = groupcounts(clean_dataset(:,2));

    [~, min_arousal] = min(samples_arousal);
    [~, max_arousal] = max(samples_arousal);

    [~, min_valence] = min(samples_valence);
    [~, max_valence] = max(samples_valence);
end
fprintf(" Balancing ended\n");

samples_arousal = groupcounts(clean_dataset(:,1));
samples_valence = groupcounts(clean_dataset(:,2));
figure("Name", "Samples for arousal after balancing");
bar(samples_arousal);
title("Samples for arousal after balancing");
fprintf("Arousal data balanced\n");
figure("Name", "Samples for valence after balancing");
bar(samples_valence);
title("Samples for valence after balancing");
fprintf("Valence data balanced\n");
save("data/clean_dataset.mat", "clean_dataset");

