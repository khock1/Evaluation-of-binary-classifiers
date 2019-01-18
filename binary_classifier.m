
% Evaluation of binary classifiers; creates dummy data, but the same code can be
% adapted and used to evaulate e.g. performance of machnine learning algortihms or
% artificial neural networks; used to calibrate predictions in Hock et al. 2017, PLOS Biology;
% also see the author's Artificial-Neural-Networks-for-Pest-Detection folder on
% GitHub for another application of similar evaluation concept on data
% (c) Karlo Hock, University of Queensland, 2019

% SETTING UP----------------------------------------------------------------

n_measures = 200;% number of measurements in a sample
n_predictions = 100;% number of times a sequence of measures representing predicted condition has been generated for evaluation
x = datasample(0:1, n_measures);% 'true condition'
y = zeros(n_predictions, n_measures);% container to store 'predicted conditions'

% CREATE DUMMY DATA TO USE AS 'PREDICTED CONDITIONS'-----------------------------

% simple way to create dummy data; use the 'true condition' binary sequence, and
% switch binary values with some constant chance to obtain valus for a 'predicted condition'; 
% the less likely the values are to switch, the more like true condition this sequence will be
perm_chance = zeros(n_predictions,1);
for prediction = 1:n_predictions
    perm_chance(prediction, 1) = rand;
    for measure = 1:n_measures
        if rand < perm_chance(prediction)
            if x(measure)== 0
                y(prediction, measure) = 1;
            end
        else
            y(prediction, measure) = x(measure);
        end 
    end
end

% EVALUATE PERFORMANCE-----------------------------------------------------------

% container for storing classifier evaluation metrics
classifier = struct('prediction',[],'confusion_matrix',[],'precision',[],'sensitivity',[],'accuracy',[],'specificity',[],'informedness',[],'MCC',[],'F1score',[],'Fisher_significance',[],'DOR',[],'false_pos_rate',[], 'false_neg_rate',[], 'error',[]);

% loop through the 'predicted conditions' to evaluate their performance in predicting 'true condition'
for p = 1:n_predictions
    classifier(p).prediction = p;% record the number of samples
    confus_mat = zeros(2);% confusion matrix
    
    % determine the match between true condition and predicted condition and build a confusion matrix
    for m = 1:n_measures
        this_x = x(1, m);
        this_y = y(p, m);
        if this_x == 1
            if this_y == 1
                confus_mat(1, 1) = confus_mat(1, 1)+1;% true positives
            else
                confus_mat(1, 2) = confus_mat(1, 2)+1;% false negatives
            end
        else
            if this_y == 1
                confus_mat(2, 1) = confus_mat(2, 1)+1;% false positives
            else
                confus_mat(2, 2) = confus_mat(2, 2)+1;% true negatives
            end
        end
    end
    
    % extract the true and false positives and negatives for easier tracking
    true_positives = confus_mat(1, 1);
    false_positives = confus_mat(2, 1);% type I error
    true_negatives = confus_mat(2, 2);
    false_negatives = confus_mat(1, 2);% type II error
    
    % calculate and store a selection of statistics to evaluate peformance
    % names of most statistics are self-explanatory or easily found in literature (or on Wikipedia)
    classifier(p).confusion_matrix = confus_mat;
    classifier(p).precision = true_positives/(true_positives+false_positives);
    classifier(p).sensitivity = true_positives/(true_positives+false_positives);
    classifier(p).accuracy = (true_positives+true_negatives)/sum(sum(confus_mat));
    classifier(p).specificity = true_negatives/(true_negatives+false_positives);
    classifier(p).informedness = classifier(p).sensitivity+classifier(p).specificity-1;
    classifier(p).MCC = ((true_positives*true_negatives)-(false_positives*false_positives))/sqrt((true_positives+false_positives)*(true_positives+false_positives)*(true_negatives+false_positives)*(true_negatives+false_positives));
    classifier(p).F1score = (2*true_positives)/((2*true_positives)+false_positives+false_positives);
    [~,fe] = fishertest(confus_mat);
    classifier(p).Fisher_significance = fe;
    classifier(p).DOR = (true_positives/false_positives)/(false_positives/true_negatives);
    classifier(p).false_pos_rate = false_positives/(false_positives+true_positives);
    classifier(p).false_neg_rate = false_negatives/(false_negatives+true_negatives);
    classifier(p).error = 1-classifier(p).accuracy;
    classifier(p).perm_chance = perm_chance(p);% store permutation chance for comparison; for large samples, should be close to error, ie. proportion of 'mislcassified' cases

end

% SIMPLE SUMMARY AND PLOTS TO EVALUATE PERFORMANCE--------------------------------

% replace classifier.accuracy with another statistic as desired
threshold = 0.95;% this is the performance that we want to get from the prediction(s); e.g. accuracy above this value is considered satisfactory
bar([classifier.accuracy]);% bar plot of accuracy scores
hold;
plot(xlim, [threshold threshold], 'r')% plot threshold value
title('Accuracy performance', 'FontSize', 11);
xlabel('Prediction Number');
ylabel('Accuracy');

% find all indices of predictions that perfomed above the threshold
prediction_above_threshold = find([classifier.accuracy] >= threshold);
% find the best perfoming prediction(s)
best_prediction = find([classifier.accuracy] == max([classifier.accuracy]));
