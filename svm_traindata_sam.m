% change the directory to where the files are and load the data
clear 
%cd C:\Users\Owner\Desktop\SCOPE
load data/test_15_matlab_data.mat

% after loading the .csv, convert the table to array of char and doubles
% table2array(labelSet)

%%

% define variable
num_feat = 8;
feat_names = ["Max Mag (t)","Abs Mean Mag (t)","Int Abs Mag (t)","RMS (t)","RMS of power (f)","STD (f)","Freq Mean (f)","Freq Median (f)"];


% choose two events to compare in classifier model
event1 = 'SPACE';
event2 = 'DOWN';

%convert char array to string array
event1 = convertCharsToStrings(event1);
event2 = convertCharsToStrings(event2);

% Put the labels into a cateogorical vector to make life easier
y = categorical(event_names);
idx_eventsofinterest = (y==event1) | (y==event2); %get the events of interest given the labels above
y = y(idx_eventsofinterest);
y = removecats(y); %This just removes extra category labels from the memory of y


X = [set0(:,idx_eventsofinterest)' set1(:,idx_eventsofinterest)']; %This puts the the two set matrices side by side, making a 415 x 16 matrix (because I also transposed them)
% X also only includes the events of interest

num_events1 = sum(y==event1)
num_events2 = sum(y==event2)


% create a histogram figure for each features
%tiledlayout(4,2) %SWM: just had to switch to subplot because i'm running
%matlab 2019a


% plot each histogram
% The original version of this was comparing the data for set0 (channel 1?)
% and set1 (channel 2), but what we're really interested in is 
for ch = 0:1 % making a separate figure for each channel
    eval(strcat('featureofinterest = set',num2str(ch),';'))
    figure
    for i = 1:num_feat
        % move to next plot
        % plot the histogram of each feature from each dataset
        %nexttile
        subplot(4,2,i) %SWM: added subplot instead of tile for version 2019a
        H = histogram(X(:,i)); % SWM: Use this to get bin sizes, but then write over it (so don't put hold on yet)
        bins = H.BinEdges; %SWM: get common bins for the two histograms
        histogram(X(y==event2,i+(ch*num_feat)),bins); %Uses the bins that are best for both sets
        hold on
        histogram(X(y==event1,i+(ch*num_feat)),bins); 
        hold off
        title(strcat('Channel ', num2str(ch +1),' - ',feat_names{i}));
    end
end

% training classifier model (SVM)


SVMModel = fitcsvm(X,y); %build the model
label = predict(SVMModel,X); %run the model on the data that you built it on (will overfit, overestimating how good it is)
figure; confusionchart(y,label)
L = loss(SVMModel,X,y);
1-L %accuracy

%%
% A second version of the model designed to penalize getting the less
% common class wrong.; %This will give lower scores, but potentially more
% transferrable results
imbalancedcostmatrix.ClassNames = unique(y); 
%imbalancedcostmatrix.ClassificationCosts =  [0 sum(y == imbalancedcostmatrix.ClassNames(2)); sum(y == imbalancedcostmatrix.ClassNames(1)) 0];
imbalancedcostmatrix.ClassificationCosts =  [0 1;3  0];

SVMModel2 = fitcsvm(X,y,'Cost',imbalancedcostmatrix,'KernelFunction','Linear'); %build the model
label2 = predict(SVMModel2,X); %run the model on the data that you built it on (will overfit, overestimating how good it is)
figure; confusionchart(y,label2)
L2 = loss(SVMModel2,X,y);
1-L2 %accuracy


% Sam's code
% trainedClassifier = fitcsvm(features(selectedepochs,:),response)

