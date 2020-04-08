% Feature selection and classification for facial surface EMG data
% Sam Michalka 4/4/2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear

%% Load the data already preprocessed and epoched

% Preprocessing for these data sets includes:
%   High pass filter at 0.5 Hz
%   Notch from 56 to 64 and 116 to 124 (for harmonic that was showing up in
%   some)
%   Epoching from -0.5 to 3.5 seconds around event. No baseline removal and
%   no epoch mean removal (so you may want to do this)

%load smilefrown1filt0p5notch56t64epochs.mat  % This one might not have the
%120 notch
%load smilefrown2filt0p5notch56t64p120epoch.mat
load run15filt0p5doublenotch56t64a120epochs.mat

%load(fullfile('C:\Users\saman\Documents\MATLAB\study1_emg', 'study1_EMG_P-01combined.mat'))

%% Add some info to the EEG structure to make life easier (trial labels)
% SKIP THIS FOR THE study1_emg version

EEG.timessec = EEG.times./1000; %version of times in seconds, useful for signal processing

% Create a new categorical variable for easier manipulation.
% Note, this will be inaccurate if you select a subset of the data without
% subsetting this.
for epoch = 1:EEG.trials
    for i = 1:length(EEG.epoch(epoch).eventlatency)
        if EEG.epoch(epoch).eventlatency{i} == 0
            EEG.epochlabelscat{epoch} = EEG.epoch(epoch).eventtype{i};
        end
    end
end
EEG.epochlabelscat = categorical(EEG.epochlabelscat);
%%

%make an array of all possible labels of events in your data
availableeventlabels = unique(EEG.epochlabelscat);

% Note: This is a good place to create new trial labels, for example, if
% you wanted to only include trials where the DOWN arrow had been pressed
% multiple times in a row.


%% Select which conditions to include in your analysis

% If you want all conditions then use [];
condnames =  {"DOWN pressed", "UP pressed"};

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

idxtrials = [];
if isempty(condnames)
    idxtrials = 1:length(EEG.epochlabelscat);
else
    for c = 1:length(condnames)
        condname = condnames{c};
        idxtrialsC = find(EEG.epochlabelscat==condname);
        idxtrials = [idxtrials idxtrialsC];
    end
end

%% Probably want to add train/test separation here
dotrainandtest = true; %If false, do train only and cross validate
% You might do false if you have very little data or if you have a separate
% test set in another file

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

ttpartgroups = removecats(EEG.epochlabelscat(idxtrials))';

if dotrainandtest     
    rng(101); %Do this to theoretically get the same
    ttpart = cvpartition(ttpartgroups,'HoldOut',.25);


    % Set up table of data that includes the trial labels
    traindata = table(EEG.epochlabelscat(idxtrials(ttpart.training))',idxtrials(ttpart.training)','VariableNames',{'labels','origEEGtrialidx'});
    traindata.labels = removecats(traindata.labels); % This remove the other trial catgories that we're not using here from the train data, which will prevent a warning

    %This table includes the original trial numbers, but we may want to remove
    %this for actually running the analysis

    testdata = table(EEG.epochlabelscat(idxtrials(ttpart.test))',idxtrials(ttpart.test)','VariableNames',{'labels','origEEGtrialidx'});
    testdata.labels = removecats(testdata.labels);
else %Otherwise all the data is training
    traindata = table(EEG.epochlabelscat(idxtrials)',idxtrials','VariableNames',{'labels','origEEGtrialidx'});
    traindata.labels = removecats(traindata.labels); % This remove the other trial catgories that we're not using here from the train data, which will prevent a warning
end
    
traindata = traindata(randperm(height(traindata)),:); % Use this if you
%have a temporal classifier so your train is not all one category first

%% Do feature extraction for selected trials

timewindowforfeatures = [2 2000]; %start and stop in ms. If timepoints don't line up, this will select a slightly later time
timewindowepochidx = (find(EEG.times>=timewindowforfeatures(1),1)):(find(EEG.times>=timewindowforfeatures(2),1));



includedfeatures = {'rms', 'var', 'iemg', 'ssi', 'mpv'}; %names of included features in the data table
includedchannels = 1:2; %channels to included, this will calculate features for each separately 
%(if you have cross channel features, you need to write something in to
%skip in order to avoid repeat features)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loop through all desired features and put into data table
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for ttround = 1:2
    if ttround == 1
        %Do training data first
        tempdata = traindata;
        if dotrainandtest
            idxt = idxtrials(ttpart.training);
        else
            idxt = idxtrials;
        end
    elseif ttround == 2
        tempdata = testdata;
        idxt = idxtrials(ttpart.test);
    end
    

    for ch = includedchannels %not necessarily a linear index, so be careful
        for f = 1:length(includedfeatures)
            clear fvalues    
            switch includedfeatures{f}
                case 'absmean'
                    td = EEG.data(ch,:,idxt) - mean(EEG.data(ch,timewindowepochidx,idxt)); %remove the trial mean
                    fvalues = squeeze(mean(abs(td),2));
                case 'rms'
                    fvalues = squeeze(rms(EEG.data(ch,timewindowepochidx,idxt)));
                case 'iemg'
                    fvalues = squeeze(sum(abs(EEG.data(ch,timewindowepochidx,idxt)), 2));
                case 'ssi'
                    fvalues = squeeze(sum(abs(EEG.data(ch,timewindowepochidx,idxt)), 2));
                case 'mpv'
                    fvalues = squeeze(max(EEG.data(ch,timewindowepochidx,idxt), [], 2));
                case 'mmav1'
                    low = prctile(EEG.data(ch,timewindowepochidx,idxt),25,2);
                    high = prctile(EEG.data(ch,timewindowepochidx,idxt),75,2);
                    weightedVals = EEG.data(ch,timewindowepochidx);
                    weightedVals(weightedVals < low) = weightedVals(weightedVals < low)*.5;
                    weightedVals(weightedVals > high) = weightedVals(weightedVals > high)*.5;
                    fvalues = squeeze(mean(abs(weightedVals),2));
                case 'var'
                    fvalues = squeeze(var(EEG.data(ch,timewindowepochidx,idxt), 0, 2));
                case 'trialstd'
                    fvalues = squeeze(std(EEG.data(ch,timewindowepochidx,idxt),0,2));
                otherwise
                    disp(strcat('unknown feature: ', includedfeatures{f},', skipping....'))
            end

            % Make sure fvalues is the right shape !!!!!!!!!!!!!!!!!
            if size(squeeze(fvalues),1) ~= length(idxt)
                warning(strcat('fvalues does not fit in data table, skipping feature: ', includedfeatures{f},...
                    '. _  Please fix the code to align shapes. Num trials: ',num2str(length(idxt)),...
                    ' and size fvalues : ',num2str(size(fvalues))))
            else

                % Put fvalues into the datatable with appropriate feature name
                eval(['tempdata.FEAT_ch' num2str(ch) '_' includedfeatures{f} ' = fvalues;']);
            end
        end
    end

    if ttround == 1
        %Do training data first
        traindata = tempdata;
        clear tempdata;
        if ~dotrainandtest % Check to see if you're only doing training data
            break;
        end
    elseif ttround == 2
        testdata = tempdata;
        clear tempdata
    end  
end

        
%% Run classification  

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
    'KernelFunction', 'linear', ...
    'Standardize',true,...
    'Cost',imbalancedcostmatrix );  % Rows are true for cost matrix

% k-fold cross validation
kval = min(5,height(response)-2); %Choose number of folds. You can also just set this manually.
cpart = cvpartition(response{:,1},'KFold',kval); % k-fold stratified cross validation
partitionedModel = crossval(trainedClassifier,'CVPartition',cpart);
[validationPredictions, validationScores] = kfoldPredict(partitionedModel);
     
% Cross validation output
validationAccuracy = 1 - kfoldLoss(partitionedModel);%, 'LossFun', 'ClassifError');
fprintf('\nValidation accuracy = %.2f%%\n', validationAccuracy*100);
trainconchart = confusionchart(traindata.labels,validationPredictions,'Normalization','row-normalized');
trainconchart.NormalizedValues

if dotrainandtest
    % Code for prediction of test data (stored here for later)
    [predictedlabel,score] = predict(trainedClassifier,testdata(:,predictorNames));
    testAccuracy = sum(testdata.labels==predictedlabel)./length(testdata.labels);
    fprintf('\nTest accuracy = %.2f%%\n', testAccuracy*100);
    testconchart = confusionchart(testdata.labels,predictedlabel,'Normalization','row-normalized');
    testconchart.NormalizedValues
end