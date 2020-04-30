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

%load data/smilefrown1filt0p5notch56t64epochs.mat  % This one might not have the
%120 notch
%load data/smilefrown2filt0p5notch56t64p120epoch.mat
%load data/run15filt0p5doublenotch56t64a120epochs.mat
%load data/run16rawdatafilt0p5notch56t64epochs.mat
%load data/smilefrownangryblinkS1filt0p5notch56t64epochs.mat
%load data/run17rawdatafilt0p5notch56t64epochs.mat

load data/run18rawdatafilt0p5notch56t64epochs.mat
%load data/run20rawdatafilt0p5notch56t64epochs.mat

%load(fullfile('C:\Users\saman\Documents\MATLAB\study1_emg', 'study1_EMG_P-01combined.mat'))

makebalanced = true;
% If you want all conditions then use [];
condnames =  {"DOWN pressed", "SPACE pressed"};

dotrainandtest = true; %If false, do train only and cross validate
% You might do false if you have very little data or if you have a separate
% test set in another file

%% Add some info to the EEG structure to make life easier (trial labels)

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

%make an array of all possible labels of events in your data
availableeventlabels = unique(EEG.epochlabelscat);

% Note: This is a good place to create new trial labels, for example, if
% you wanted to only include trials where the DOWN arrow had been pressed
% multiple times in a row.


%% Do additional preprocessing: filter, remove mean or baseline from trials 

for channel =1:size(EEG.data,1)
    % First, save the power in the frequency comain
    %[EEG.freqcalcs.P(channel,:,:),EEG.freqcalcs.F] = pspectrum(squeeze(EEG.data(channel,:,:)),EEG.timessec);
    
    %Trial by trial baseline removal (can also do pre-event baseline removal
    %instead
    EEG.data(channel,:, :) = EEG.data(channel,:, :) - mean(EEG.data(channel,:, :),2);
    
    % additional filtering (might want to do this with the continuous data
    % instead, if you find somethign you like)
    %EEG.data(channel,:, :) = highpass(squeeze(EEG.data(channel,:, :)),20,EEG.srate);
end

%% Select which conditions to include in your analysis and if you want to balance your data

if makebalanced
    
    mydata = EEG.data;
    
    for c = 1:length(condnames)
        num_trials(c) = sum(EEG.epochlabelscat==condnames{c});
    end
    num_trialspercond = min(num_trials);

    clear idx bicx balidx
    for c = 1:length(condnames)
        idx{c,:} = find(EEG.epochlabelscat==condnames{c});
        bicx(c,:) = idx{c}(randperm(length(idx{c}),num_trialspercond));
    end

    balidx = reshape(bicx,[],1);
    balidx = balidx(randperm(length(balidx)));

    EEG.data = mydata(:,:,balidx);
   
    EEG.epochlabelscat = EEG.epochlabelscat(balidx);
    
end


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

%% Train/test separation 


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
    



%% Do feature extraction for selected trials

w.totaltimewindow = [0 2000]; %start and stop in ms. If timepoints don't line up, this will select a slightly later time
%timetimewindowepochidx = (find(EEG.times>=timewindowforfeatures(1),1)):(find(EEG.times>=timewindowforfeatures(2),1));

w.timewindowbinsize = 200; %This should ideally divide into an equal number of time points
w.timewindowoverlap = 0; %128 for another paper

w.starttimes = w.totaltimewindow(1):(w.timewindowbinsize-w.timewindowoverlap):w.totaltimewindow(2);
w.endtimes = (w.totaltimewindow(1)-1+w.timewindowbinsize):(w.timewindowbinsize-w.timewindowoverlap):w.totaltimewindow(2);
if w.totaltimewindow(2) - w.starttimes(end) <= w.timewindowoverlap %if increment is smaller than the overlap window
    w.starttimes(end) = []; %then remove the last one (avoids indexing problem, plus you've already used this data)
end
if length(w.starttimes) > length(w.endtimes)
    w.endtimes = [w.endtimes w.totaltimewindow(2)];
    warning('The timewindowbinsize does not split evenly into the totaltimewindow, last window will be smaller')
    w.endtimes - w.starttimes
end
w.alltimewindowsforfeatures = [w.starttimes; w.endtimes]; %(:,1) for first pair

%% Select features

%includedfeatures = {'bp2t20','bp40t56','bp64t80' ,'bp80t110'};
includedfeatures = {'bp2t20','bp20t40','bp40t56','bp64t80' ,'bp80t110','rms', 'iemg','mmav1','var','mpv','var','ssi'};
% includedfeatures = {'bp40t56','bp64t80' ,'bp80t110','rms', 'iemg','mmav1','var'};
%includedfeatures = {'rms','ssi'}%, 'absmean','ssi','iemg','mmav1','mpv','var'}; %names of included features in the data table
%includedfeatures = {'rms', 'iemg','mmav1','var'}; %names of included features in the data table
%includedfeatures = {'bp40t56','bp64t80' ,'bp80t110','rms', 'iemg','mmav1','var', 'medianfreq'};
% includedfeatures = {'rms', 'absmean','ssi','iemg','mmav1','mpv','var'}; %names of included features in the data table
includedfeatures = {'rms', 'iemg','mmav1','var', 'mfl', 'wamp'}; % Include mfl and wamp
% includedfeatures = {'bp40t56','bp64t80' ,'bp80t110','rms', 'iemg','mmav1','var', 'medianfreq'};
includedchannels = 1:2; %channels to included, this will calculate features for each separately 
%(if you have cross channel features, you need to write something in to
%skip in order to avoid repeat features)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Loop through all desired features and put into data table
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for ttround = 1:2
    % Deal with sometimes training only and sometimes training and test
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
    
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Add features to the data table
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for ch = includedchannels %not necessarily a linear index, so be careful
        for f = 1:length(includedfeatures)
            fvalues =[]; %clear and initialize fvalues
            for  tw = 1:size(w.alltimewindowsforfeatures,2)
                timewindowforfeatures = w.alltimewindowsforfeatures(:,tw);
                timewindowepochidx = (find(EEG.times>=timewindowforfeatures(1),1)):(find(EEG.times>=timewindowforfeatures(2),1));
                %EEG.times(timewindowepochidx(end)) -EEG.times(timewindowepochidx(1))
                
                switch includedfeatures{f}
                    case 'absmean'
                        %td = EEG.data(ch,:,idxt) - mean(EEG.data(ch,timewindowepochidx,idxt)); %remove the trial mean
                        fvalues = [fvalues squeeze(mean(abs(EEG.data(ch,:,idxt)),2))];
                    case 'rms'
                        fvalues = [fvalues squeeze(rms(EEG.data(ch,timewindowepochidx,idxt)))];                   
                    case 'iemg'
                        fvalues = [fvalues squeeze(sum(abs(EEG.data(ch,timewindowepochidx,idxt)), 2))];
                    case 'ssi'
                        fvalues = [fvalues squeeze(sum(abs(EEG.data(ch,timewindowepochidx,idxt)), 2))];
                        % This could instead be done with the integral, which gives a smaller but correlated number if rectified
                        %ctr = squeeze(cumtrapz(EEG.timessec(timewindowepochidx), abs(EEG.data(ch,timewindowepochidx,idxt))));
                        %fvalues = ctr(end,:)'; % This could be useful if you
                        %wanted to take the diff between different increments.
                    case 'mpv'
                        fvalues = [fvalues squeeze(max(EEG.data(ch,timewindowepochidx,idxt), [], 2))];
                    case 'mmav1'
                        low = prctile(EEG.data(ch,timewindowepochidx,idxt),25,2);
                        high = prctile(EEG.data(ch,timewindowepochidx,idxt),75,2);
                        weightedVals = EEG.data(ch,timewindowepochidx,idxt); %SWM: added idxt
                        weightedVals(weightedVals < low) = weightedVals(weightedVals < low)*.5;
                        weightedVals(weightedVals > high) = weightedVals(weightedVals > high)*.5;
                        fvalues = [fvalues squeeze(mean(abs(weightedVals),2))];
                    case 'var'
                        fvalues = [fvalues squeeze(var(EEG.data(ch,timewindowepochidx,idxt), 0, 2))];
                    case 'bp2t20'                       
                        fvalues = [fvalues bandpower(squeeze(EEG.data(ch,timewindowepochidx,idxt)),EEG.srate,[2 20])'];
                    case 'bp20t40'                       
                        fvalues = [fvalues bandpower(squeeze(EEG.data(ch,timewindowepochidx,idxt)),EEG.srate,[20 40])'];
                    case 'bp40t56'                       
                        fvalues = [fvalues bandpower(squeeze(EEG.data(ch,timewindowepochidx,idxt)),EEG.srate,[40 56])'];
                    case 'bp64t80'                       
                        fvalues = [fvalues bandpower(squeeze(EEG.data(ch,timewindowepochidx,idxt)),EEG.srate,[64 80])'];
                    case 'bp80t110'                       
                        fvalues = [fvalues bandpower(squeeze(EEG.data(ch,timewindowepochidx,idxt)),EEG.srate,[80 110])'];
                    case 'medianfreq'
                        fvalues = [fvalues squeeze(real(median(fft(EEG.data(ch,timewindowepochidx,idxt), '', 2), 2)))];
                    case 'mfl'
                        fvalues = [fvalues squeeze(real(log10(sqrt(sum(diff(EEG.data(ch,timewindowepochidx,idxt)).^2, 2)))))];
                    case 'wamp'
                        % There is almost definitely a better way to do
                        % this.
                        threshold = 0.05;
                        shifted = circshift(EEG.data(ch,timewindowepochidx, idxt), 1, 2);
                        wamp_sum = sum(abs(EEG.data(ch,timewindowepochidx, idxt)) + threshold < (abs(shifted)), 2);
                        fvalues = [fvalues squeeze(wamp_sum)];
                    otherwise
                        disp(strcat('unknown feature: ', includedfeatures{f},', skipping....'))
                end
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
        %traindata = tempdata;
        
        % Randomly permute the data so not ordered
        traindata = tempdata(randperm(height(tempdata)),:);
        clear tempdata;
        if ~dotrainandtest % Check to see if you're only doing training data
            break;
        end
    elseif ttround == 2
        %testdata = tempdata;
        
        %Randomly permute the test data
        testdata = tempdata(randperm(height(tempdata)),:);
        
        clear tempdata
    end  
end

        
%% Run classification  

traindata = splitvars(traindata); %split subvariables into independent variable to not anger the classifier.

fprintf(strcat('Classifying _',condnames{1}, ' and _', condnames{2},' \n'))
% Predictors are features
predictorNames = traindata.Properties.VariableNames(3:end); %This selects all but first two (assumed to be labels and orig index number, fix if this is not the case, or you can manually select here)
predictors = traindata(:,predictorNames); %This
response = traindata(:,'labels'); %labels

imbalancedcostmatrix.ClassNames = unique(response{:,1}); 
imbalancedcostmatrix.ClassificationCosts =  [0 sum(traindata.labels == imbalancedcostmatrix.ClassNames(2)); sum(traindata.labels == imbalancedcostmatrix.ClassNames(1)) 0];
%imbalancedcostmatrix.ClassificationCosts
trainedClassifier = fitcsvm(predictors, ...
    response, ...
    'KernelFunction', 'Linear', ...
    'Standardize',true);%,...  
    %'Prior','empirical');%,...
    %'Cost',imbalancedcostmatrix );  % Rows are true for cost matrix
   
%trainedClassifier = fitcecoc(predictors,response,'KernelFunction', 'Linear');
%'OutlierFraction',0.15,...

% k-fold cross validation
kval = 5;%min(5,height(response)-2); %Choose number of folds. You can also just set this manually.
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

% if dotrainandtest
%     testdata = splitvars(testdata); %split subvariables into variables
%     
%     % Code for prediction of test data (stored here for later)
%     [predictedlabel,score] = predict(trainedClassifier,testdata(:,predictorNames));
%     testAccuracy = sum(testdata.labels==predictedlabel)./length(testdata.labels);
%     fprintf('\nTest accuracy = %.2f%%\n', testAccuracy*100);
%     figure;
%     testconchart = confusionchart(testdata.labels,predictedlabel);%,'Normalization','row-normalized'
%     testconchart.NormalizedValues
% end