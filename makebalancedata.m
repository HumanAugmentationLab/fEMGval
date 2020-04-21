% Make balanced data

mydata = run17train;
condnames =  {"SPACE pressed", "DOWN pressed"};

for c = 1:length(condnames)
    num_trials(c) = sum(mydata.labels==condnames{c});
end
num_trialspercond = min(num_trials);


for c = 1:length(condnames)
    
    idx{c,:} = find(mydata.labels==condnames{c});
    bicx(c,:) = idx{c}(randperm(length(idx{c}),num_trialspercond));
    
end

balidx = reshape(bicx,[],1);
balidx = balidx(randperm(length(balidx)));

mybaldata = mydata(balidx,:);

run17bal = mybaldata;

%% Run classification on balanced data

load data\run17and18balanced.mat

traindata = run18bal;
testdata = run17bal;

dotrainandtest = true;

traindata = splitvars(traindata); %split subvariables into independent variable to not anger the classifier.



fprintf(strcat('Classifying _',condnames{1}, ' and _', condnames{2},' \n'))
% Predictors are features
predictorNames = traindata.Properties.VariableNames(3:end); %This selects all but first two (assumed to be labels and orig index number, fix if this is not the case, or you can manually select here)
predictors = traindata(:,predictorNames); %This
response = traindata(:,'labels'); %labels

imbalancedcostmatrix.ClassNames = unique(response{:,1}); 
imbalancedcostmatrix.ClassificationCosts =  [0 sum(traindata.labels == imbalancedcostmatrix.ClassNames(2)); sum(traindata.labels == imbalancedcostmatrix.ClassNames(1)) 0];
imbalancedcostmatrix.ClassificationCosts
trainedClassifier = fitcsvm(predictors, ...
    response, ...
    'KernelFunction', 'Polynomial', ...
     'Standardize',true,...
    'OutlierFraction',0.1);
% 
%     'KernelFunction', 'Polynomial', ...
%     'Standardize',true,...
%     'Optimize
%     'OutlierFraction',0.1);%,...
    %'Cost',imbalancedcostmatrix );  % Rows are true for cost matrix 'OptimizeHyperparameters','all',...
   
%trainedClassifier = fitcecoc(predictors,response,'KernelFunction', 'Linear');
%'OutlierFraction',0.15,...

% k-fold cross validation
kval = min(5,height(response)-2); %Choose number of folds. You can also just set this manually.
cpart = cvpartition(response{:,1},'KFold',kval); % k-fold stratified cross validation
partitionedModel = crossval(trainedClassifier,'CVPartition',cpart);
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);
    
% Cross validation output
validationAccuracy = 1 - kfoldLoss(partitionedModel);%, 'LossFun', 'ClassifError');
fprintf('\nValidation accuracy = %.2f%%\n', validationAccuracy*100);
trainconchart = confusionchart(traindata.labels,validationPredictions);
% ,'Normalization','row-normalized'
% 
trainconchart.NormalizedValues

% Try to calculate the prob of this from chance classifier; not sure this
% is correct, need to think through.
mydist= makedist('Binomial',length(traindata.labels),0.5);
fprintf('\n Proportion of .5 classifiers that would do worse than this is %.4f% \n',mydist.cdf(sum(validationPredictions==traindata.labels)))

if dotrainandtest
    testdata = splitvars(testdata); %split subvariables into variables
    
    % Code for prediction of test data (stored here for later)
    [predictedlabel,score] = predict(trainedClassifier,testdata(:,predictorNames));
    testAccuracy = sum(testdata.labels==predictedlabel)./length(testdata.labels);
    fprintf('\nTest accuracy = %.2f%%\n', testAccuracy*100);
    figure;
    testconchart = confusionchart(testdata.labels,predictedlabel);%,'Normalization','row-normalized'
    testconchart.NormalizedValues
    
    % Try to calculate the prob of this from chance classifier
    mydist= makedist('Binomial',length(testdata.labels),0.5);
    fprintf('\n Proportion of .5 classifiers that would do worse than this is %.4f% \n',mydist.cdf(sum(predictedlabel==testdata.labels)))

end
fprintf('\n')