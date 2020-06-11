% Analysis of Valve data (Sam Michalka 4/2020)
clear


allruns = [15 17 18 20];
for r= 1:length(allruns)
    clearvars testSet0 testSet1 idx bicx X y
    %opnumfeat = [17,11]; %run, number

    % Code for doing from Bryans folder system
    runnum = allruns(r);
    dirname = 'data/studentfeatures/extended/';
    testSet0 = load(strcat(dirname,'testSet0_',num2str(runnum),'new_relaxedDown.csv'));
    testSet1 = load(strcat(dirname,'testSet1_',num2str(runnum),'new_relaxedDown.csv'));
%      testSet0 = load(strcat(dirname,'testSet0_',num2str(runnum),'new.csv'));
%      testSet1 = load(strcat(dirname,'testSet1_',num2str(runnum),'new.csv'));
    lcell = importdata(strcat(dirname,'labelSet_',num2str(runnum),'new_relaxedDown.csv'));
    lcell = split(lcell{1,1},',',1);
    labels = categorical(lcell);
    data = [testSet0 ; testSet1];
    %data = [testSet0];
    num_channels =2;

    % Code for doing from Minjus folder system
    % dirname = 'data/studentfeatures/trial_15/';
    % load(strcat(dirname,'testSet0.csv'));
    % load(strcat(dirname,'testSet1.csv'));
    % lcell = importdata(strcat(dirname,'labelSet.csv'));
    % lcell = split(lcell{1,1},',',1);
    % labels = categorical(lcell);
    % data = [testSet0; testSet1];

    %Version to read from .mat files that Minju wrote, but now just parsing
    %from the original features by Bryan.
    % clear
    % load data/test_19_matlab_data.mat
    % 
    % labels = categorical(event_names)';
    % data = [set0; set1];
    % clear set0 set1 event_names

    %% Select only the conditions you want, remove the rest
    condnames = {'S Pressed','F Pressed'};
    idxdata = (labels== condnames{1} | labels ==condnames{2});

    y = labels(idxdata);
    y = removecats(y);
    X = data(:,idxdata)';

    %% Make the data balanced by undersampling (you can skip this step if you want imbalanced data)
    Xorig = X;
    yorig = y;
    num_trials=[];
    for c = 1:length(condnames)
        num_trials(c) = sum(y==condnames{c});
    end
    num_trialspercond = min(num_trials); %Figure out which condition has fewer trials


    for c = 1:length(condnames)
        idx{c,:} = find(y==condnames{c}); %all labels
        bicx(c,:) = idx{c}(randperm(length(idx{c}),num_trialspercond)); %random selection of these
    end

    balidx = reshape(bicx,[],1);
    balidx = balidx(randperm(length(balidx)));

    X = Xorig(balidx,:);
    y = yorig(balidx,:);


    %% Partition data into train and test

    rng(101); %Do this to theoretically get the same
    ttpart = cvpartition(y,'HoldOut',.25);

    % Set up table of data that includes the trial labels
    ytrain = y(ttpart.training);
    Xtrain = X(ttpart.training,:);

    %This table includes the original trial numbers, but we may want to remove
    %this for actually running the analysis 
    ytest = y(ttpart.test);
    Xtest = X(ttpart.test,:);
    
  
    %% Run training classification

    % Imbalanced 
    imbalancedcostmatrix.ClassNames = unique(ytrain); 
    %imbalancedcostmatrix.ClassificationCosts =  [0 sum(label == imbalancedcostmatrix.ClassNames(2)); sum(label == imbalancedcostmatrix.ClassNames(1)) 0];
    %imbalancedcostmatrix.ClassificationCosts = [0 1; 6 0];

    trainedClassifier = fitcsvm(Xtrain, ...
        ytrain, ...
        'KernelFunction', 'Linear', ...
        'Standardize',true);%,...  
        %'Cost',imbalancedcostmatrix ); 
        %'Prior','empirical');%,...
        %'Cost',imbalancedcostmatrix );  % Rows are true for cost matrix

        % k-fold cross validation
    kval = 5; 
    cpart = cvpartition(ytrain,'KFold',kval); % k-fold stratified cross validation
    partitionedModel = crossval(trainedClassifier,'CVPartition',cpart);
    [validationPredictions, validationScores] = kfoldPredict(partitionedModel);

    % Cross validation output
    %validationAccuracy = 1 - kfoldLoss(partitionedModel);%, 'LossFun', 'ClassifError');
%     valCC(r).trainconchart = confusionchart(ytrain,validationPredictions);

    validationAccuracy(r) = sum(ytrain==validationPredictions)./length(ytrain);
    fprintf('\nValidation accuracy = %.2f%%\n', validationAccuracy*100);

    % ,'Normalization','row-normalized'
    % 
%    valCC(r).trainconchart.NormalizedValues

    %% Loop through each feature
    numindfeatures = (size(X,2)/2); %SWM: FIX for 1
    if num_channels == 2
        featurepairs = [1:numindfeatures; (numindfeatures+1):(numindfeatures+numindfeatures)];
    else
        featurepairs = 1:numindfeatures;
    end
    %indfeatclassrate = zeros(numindfeatures,1);
    for nf = 1:numindfeatures
        trainedClassifierInd = fitcsvm(Xtrain(:,featurepairs(:,nf)), ...
        ytrain, ...
        'KernelFunction', 'Linear', ...
        'Standardize',true);

        partitionedModelInd = crossval(trainedClassifierInd,'CVPartition',cpart);
        [validationPredictionsInd, ~] = kfoldPredict(partitionedModelInd);
        % Record classification rates for each 
       indfeatclassrate(r,nf) =  sum(ytrain==validationPredictionsInd)./length(ytrain); 

    end

    %% Select the top f features and run the model again with just those
    for f = 1:numindfeatures

        numtopfeatures = f; %Actual number of features will be twice this

        [B,indtopfeatures] = maxk(indfeatclassrate(r,:),numtopfeatures);

        if num_channels ==2
            topfeaturepairs = [indtopfeatures (numindfeatures + indtopfeatures)];
        else
            topfeaturepairs = [indtopfeatures];
        end

        trainedClassifierInd = fitcsvm(Xtrain(:,topfeaturepairs), ...
            ytrain, ...
            'KernelFunction', 'Linear', ...
            'Standardize',true);

        partitionedModelInd = crossval(trainedClassifierInd,'CVPartition',cpart);
        [validationPredictionsInd, ~] = kfoldPredict(partitionedModelInd);

        accuracyofselfeatures(r,f) = sum(ytrain==validationPredictionsInd)./length(ytrain); 
        %fprintf('\n Selected feature accuracy = %.2f%%\n', accuracyofselfeatures*100);

    end


    %% Select the top number of features as best classifier for this dataset
%     [mm, mi] = max(accuracyofselfeatures(r,:));
%     fprintf('\n Selected feature accuracy = %.2f%% found with %i features.\n', mm*100,mi)
% 
%     numtopfeatures = mi; %Actual number of features will be twice this
% 
%     [B,indtopfeatures] = maxk(indfeatclassrate(r,:),numtopfeatures);

% Try to find average accuracy

    %indtopfeatures = [ 14    11    15    20    16     5    10    22     2    21     4     8    12    23 ];
    %indtopfeatures = [4    12    11    14    13     2     3    23    21     7]; % median across runs
    indtopfeatures = [4    12    11    14    13     2     3    23  21];
    
    %indtopfeatures = [12    11    23    14     4    13     3    18    17     2]; % mean across runs

    
    numtopfeatures = length(numtopfeatures);
    
    if num_channels == 2 %if using both channels
        topfeaturepairs = [indtopfeatures (numindfeatures + indtopfeatures)];
    else
        topfeaturepairs = [indtopfeatures];
    end

    
    
    trainedClassifierInd = fitcsvm(Xtrain(:,topfeaturepairs), ...
        ytrain, ...
        'KernelFunction', 'Linear', ...
        'Standardize',true);

    partitionedModelInd = crossval(trainedClassifierInd,'CVPartition',cpart);
    [validationPredictionsInd, ~] = kfoldPredict(partitionedModelInd);

    accuracyoftopselesctedValidation(r,1) = sum(ytrain==validationPredictionsInd)./length(ytrain); 
%    accuracyoftopselesctedValidation(r,2) = mi;




    %% Run on test data, only run this at the end after you're done tweaking parameters
    % % Code for prediction of test data (stored here for later)

    [predictedlabel,score] = predict(trainedClassifier,Xtest);
    testAccuracy(r) = sum(ytest==predictedlabel)./length(ytest);
    fprintf('\nTest accuracy = %.2f%%\n', testAccuracy*100);
%     figure;
%     testCC(r).conchart = confusionchart(ytest,predictedlabel);%,'Normalization','row-normalized'
%     testCC(r).conchart.NormalizedValues


    %% Run selected on test data, only run this at the end after you're done tweaking parameters

    [predictedlabel,score] = predict(trainedClassifierInd,Xtest(:,topfeaturepairs));
    testAccuracySelected(r) = sum(ytest==predictedlabel)./length(ytest);
    fprintf('\nTest accuracy on selected features = %.2f%%\n', testAccuracy*100);
%     figure;
%     testselCC(r).conchart = confusionchart(ytest,predictedlabel);%,'Normalization','row-normalized'
%     testselCC(r).conchart.NormalizedValues


end
%% Which ones are top
% clearvars itf
% nfth = 7;
% for r= 1:length(allruns)
%     [B,itf(r,:)] = maxk(indfeatclassrate(r,:),nfth);
% end
% 
% itf;
% figure
% histogram(itf)
% set(gca,'FontSize',14)
% xlabel('Feature Number'); ylabel('Number of runs feature in top'); 
% title(['Top ',num2str(nfth),' features per run']);

%% Plot accuracy based on number of features
figure
plot(1:numindfeatures,accuracyofselfeatures,'LineWidth',2);
xlabel('Number of features per channel'); ylabel('Classification Accuracy on Selected Features'); %ylim([0.5 .8]);
set(gca,'FontSize',14);

for r= 1:length(allruns)
    runlegend{r} = ['Run ' num2str(allruns(r))];
end
legend(runlegend);

indfeatclassrate;

%% Spit out results of accuracy

disp('train accuracy all')
validationAccuracy'
disp('test accuracy all')
testAccuracy'

%accuracyoftopselesctedValidation(:,2)
disp('train accuracy sel')
accuracyoftopselesctedValidation(:,1)
disp('test accuracy sel')
testAccuracySelected'


disp('mean accuracies')
mean(accuracyoftopselesctedValidation(:,1))


%% Plot of the accuracy of individual features

figure;
%bar(indfeatclassrate)
bar(mean(indfeatclassrate))
hold on
errorbar(mean(indfeatclassrate), 2*std(indfeatclassrate),'Color','k', 'LineWidth',2,'LineStyle','none');

ylim([0.2, 0.8]);
yline(0.5,'k--');
ylabel('Classification Accuracy');
xlabel('Feature Number')
title('Classification Rates for Individual Features (2 channels)')
set(gca,'FontSize',14)

%% Plot accuracy of individual features as line graph
figure
plot(indfeatclassrate','LineWidth',2);
hold on
plot(mean(indfeatclassrate),'k--','LineWidth',2);
runlegend{5} ="Average";
runlegend{6} = "Chance";

ylim([0.2, 0.8]);
yline(0.5,'k--');
legend(runlegend);
ylabel('Classification Accuracy');
xlabel('Feature Number')
title('Classification Rates for Individual Features (2 channels)')
set(gca,'FontSize',18)
%% median
% mean(testAccuracy)
% 
% ans =
% 
%     0.6306
% 
% mean(testAccuracySelected)
% 
% ans =
% 
%     0.7326
% 
% mean(validationAccuracy)
% 
% ans =
% 
%     0.6099
% 
% mean(accuracyoftopselesctedValidation(:,1))
% 
% ans =
% 
%     0.5901
% 


% % with 8
% mean(accuracyoftopselesctedValidation(:,1))
% 
% ans =
% 
%     0.6110
% 
% mean(validationAccuracy)
% 
% ans =
% 
%     0.6099
% 
% mean(testAccuracySelected)
% 
% ans =
% 
%     0.7247
% 

%%
% 
% % for baseline vs space
% % Top features from all
% avgfeaturestop = [ 14    11    15    20    16     5    10    22     2    21     4     8    12    23 ];
% topfeatures7 = [11 14 15 20]; %best
% topfeatures14 = [ 1:5 10 14:17 20];
% 
% % top features from 17 18
% topfeature7for1718 = [15 20];
% topfeature14for1718 = [3 10 14 15 17 20 21 23]; % best
% 

% % for baseline vs down
% % top features from all
% % avgfeaturesstop = [22    23    21     5     3    20    17     9    10
% 7     6    11     1    12    18];
% %topfeatures14 = [3 5 10 21 22 23] then 6 7 12 17 20

% selfeatures = topfeatures7
% allruns = [15 17 18 19];
% for r= 1:length(allruns)
%     clearvars testSet0 testSet1 idx bicx X y
%     %opnumfeat = [17,11]; %run, number
% 
%     % Code for doing from Bryans folder system
%     runnum = allruns(r);
%     dirname = 'data/studentfeatures/extended/';
%     testSet0 = load(strcat(dirname,'testSet0_',num2str(runnum),'new.csv'));
%     testSet1 = load(strcat(dirname,'testSet1_',num2str(runnum),'new.csv'));
%     lcell = importdata(strcat(dirname,'labelSet_',num2str(runnum),'new.csv'));
%     lcell = split(lcell{1,1},',',1);
%     labels = categorical(lcell);
%     data = [testSet0; testSet1];
% 
% 
%     % Select only the conditions you want, remove the rest
%     condnames = {'BASELINE','SPACE'};
%     idxdata = (labels== condnames{1} | labels ==condnames{2});
% 
%     y = labels(idxdata);
%     y = removecats(y);
%     X = data(:,idxdata)';
% 
%     % Make the data balanced by undersampling (you can skip this step)
%     Xorig = X;
%     yorig = y;
%     num_trials=[];
%     for c = 1:length(condnames)
%         num_trials(c) = sum(y==condnames{c});
%     end
%     num_trialspercond = min(num_trials); %Figure out which condition has fewer trials
% 
% 
%     for c = 1:length(condnames)
%         idx{c,:} = find(y==condnames{c}); %all labels
%         bicx(c,:) = idx{c}(randperm(length(idx{c}),num_trialspercond)); %random selection of these
%     end
% 
%     balidx = reshape(bicx,[],1);
%     balidx = balidx(randperm(length(balidx)));
% 
%     X = Xorig(balidx,:);
%     y = yorig(balidx,:);
% 
% 
%     % Partition data into train and test
% 
%     rng(11); %Do this to theoretically get the same
%     ttpart = cvpartition(y,'HoldOut',.25);
% 
%     % Set up table of data that includes the trial labels
%     ytrain = y(ttpart.training);
%     Xtrain = X(ttpart.training,:);
% 
%     %This table includes the original trial numbers, but we may want to remove
%     %this for actually running the analysis 
%     ytest = y(ttpart.test);
%     Xtest = X(ttpart.test,:);
% 
%     trainedClassifier = fitcsvm(Xtrain(:,selfeatures), ...
%         ytrain, ...
%         'KernelFunction', 'Linear', ...
%         'Standardize',true);%,...  
%         %'Cost',imbalancedcostmatrix ); 
%         %'Prior','empirical');%,...
%         %'Cost',imbalancedcostmatrix );  % Rows are true for cost matrix
% 
%         % k-fold cross validation
%     kval = 5; 
%     cpart = cvpartition(ytrain,'KFold',kval); % k-fold stratified cross validation
%     partitionedModel = crossval(trainedClassifier,'CVPartition',cpart);
%     [validationPredictions, validationScores] = kfoldPredict(partitionedModel);
% 
%     % Cross validation output
%     trainconchart = confusionchart(ytrain,validationPredictions);
% 
%     validationAccuracy = sum(ytrain==validationPredictions)./length(ytrain);
%     fprintf('\nValidation accuracy = %.2f%%\n', validationAccuracy*100);
%     
%     % Test data
%     [predictedlabel,score] = predict(trainedClassifier,Xtest(:,selfeatures));
%     testAccuracy = sum(ytest==predictedlabel)./length(ytest);
%     fprintf('\nTest accuracy on selected features = %.2f%%\n', testAccuracy*100);
%     
% end



