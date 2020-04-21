% change the directory to where the files are and load the data
clear all
%cd C:\Users\Owner\Desktop\SCOPE
load data/test_15_matlab_data.mat

% after loading the .csv, convert the table to array of char and doubles
% table2array(labelSet)

% define variable
num_feat = 8;
feat_names = ["Max Mag (t)","Abs Mean Mag (t)","Int Abs Mag (t)","RMS (t)","RMS of power (f)","STD (f)","Freq Mean (f)","Freq Median (f)"];

% choose two events to compare in classifier model
event1 = 'SPACE';
event2 = 'BASELINE';

%convert char array to string array
event1 = convertCharsToStrings(event1);
event2 = convertCharsToStrings(event2);

figure
% create a histogram figure for each features
%tiledlayout(4,2)


% plot each histogram
for i = 1:num_feat
    % move to next plot
    % plot the histogram of each feature from each dataset
    %nexttile
    subplot(4,2,i)
    histogram(set0(i,1:length(set0)))
    hold on
    histogram(set1(i,1:length(set1)))
    hold off
    title(feat_names(i))
end

% training classifier model (SVM)

% obtain indexes for each event
idx_event1 = ~strcmp(event_names, event1);
idx_event2 = ~strcmp(event_names, event2);

X = set0(1,idx_event1);
X = X';
y = event_names(idx_event1);

SVMModel = fitcsvm(X,y)
% Sam's code
% trainedClassifier = fitcsvm(features(selectedepochs,:),response)

